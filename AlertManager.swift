//
//  AlertManager.swift
//  Astrateq Vision
//
//  Actor-isolated real-time alerting engine. Converts `DriverState`
//  transitions from `EyeAspectCalculator` into escalating audio + haptic
//  feedback. Audio and haptic assets are pre-buffered/pre-warmed at
//  `configure()` time so there is zero decode/spin-up latency at the moment
//  a microsleep is detected — the one scenario where latency is a safety
//  issue, not just a UX one.
//
//  Requires: iOS 17+, Swift 6 (strict concurrency)
//

import Foundation
import AVFoundation
import CoreHaptics
import UIKit
import os.log

// MARK: - Errors

enum AlertManagerError: Error, LocalizedError, Sendable {
    case audioSessionConfigurationFailed(String)
    case audioAssetNotFound(String)
    case audioAssetLoadFailed(String)
    case audioEngineStartFailed(String)
    case hapticEngineStartFailed(String)

    var errorDescription: String? {
        switch self {
        case .audioSessionConfigurationFailed(let detail):
            return "Failed to configure AVAudioSession: \(detail)"
        case .audioAssetNotFound(let detail):
            return "Alert audio asset not found: \(detail)"
        case .audioAssetLoadFailed(let detail):
            return "Failed to load alert audio asset: \(detail)"
        case .audioEngineStartFailed(let detail):
            return "AVAudioEngine failed to start: \(detail)"
        case .hapticEngineStartFailed(let detail):
            return "CHHapticEngine failed to start: \(detail)"
        }
    }
}

// MARK: - Configuration

struct AlertConfiguration: Sendable {
    /// Short, gentle chime played once (plus periodic reminders) while drowsy.
    var chimeResourceName: String = "drowsy_chime"
    var chimeResourceExtension: String = "caf"

    /// Loud, looping siren played continuously during a microsleep alarm.
    var sirenResourceName: String = "microsleep_siren"
    var sirenResourceExtension: String = "caf"

    var drowsyHapticIntensity: Float = 0.4
    var drowsyHapticSharpness: Float = 0.3
    /// How often to re-issue the gentle drowsy nudge if the state persists,
    /// so a sustained low-grade drowsiness isn't only announced once.
    var drowsyReminderInterval: TimeInterval = 20

    var microsleepHapticIntensity: Float = 1.0
    var microsleepHapticSharpness: Float = 1.0
    /// Spacing between pulses in the looping microsleep haptic pattern.
    var microsleepPulseSpacing: TimeInterval = 0.25

    static let `default` = AlertConfiguration()
}

// MARK: - AlertManager

/// Actor-isolated owner of the audio and haptic alerting stack. All mutable
/// engine/player state is confined here; `handle(state:)` is the single
/// entry point consumers call with each new fatigue classification.
actor AlertManager {

    // MARK: Configuration & state

    private let configuration: AlertConfiguration
    private let logger = Logger(subsystem: "com.astrateq.vision", category: "AlertManager")
    private let hapticsSupported: Bool

    private var isConfigured = false
    private var currentState: DriverState = .alert
    private var drowsyReminderTask: Task<Void, Never>?

    // MARK: Audio

    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private var chimeBuffer: AVAudioPCMBuffer?
    private var sirenBuffer: AVAudioPCMBuffer?

    // MARK: Haptics

    private var hapticEngine: CHHapticEngine?
    private var drowsyPatternPlayer: CHHapticPatternPlayer?
    private var microsleepPatternPlayer: CHHapticAdvancedPatternPlayer?

    // MARK: Notification observers

    private var interruptionObserver: NSObjectProtocol?
    private var routeChangeObserver: NSObjectProtocol?

    // MARK: - Init

    init(configuration: AlertConfiguration = .default) {
        self.configuration = configuration
        self.hapticsSupported = CHHapticEngine.capabilitiesForHardware().supportsHaptics
    }

    deinit {
        drowsyReminderTask?.cancel()
    }

    // MARK: - Configuration / Pre-warming

    /// Configures the audio session, pre-loads all alert audio into memory,
    /// starts the audio engine, and starts + pre-warms the haptic engine.
    /// Call once during app/feature startup — well before the first fatigue
    /// assessment arrives — so `handle(state:)` never pays a cold-start cost.
    func configure() throws {
        guard !isConfigured else { return }

        try configureAudioSession()
        try loadAudioAssets()
        try prepareAudioEngine()

        if hapticsSupported {
            try prepareHapticEngine()
        } else {
            logger.warning("Device does not support CoreHaptics; falling back to UIFeedbackGenerator for alerts.")
        }

        isConfigured = true
        logger.info("AlertManager configured and pre-warmed.")
    }

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        do {
            // .playback ignores the ring/silent switch and continues even if
            // the device is muted — required so a microsleep alarm can never
            // be silenced by an incidental mute-switch flick. .duckOthers
            // lowers any concurrently playing audio (music/podcasts/nav) so
            // the alert is clearly audible without fully interrupting it.
            try session.setCategory(.playback, mode: .default, options: [.duckOthers])
            try session.setActive(true, options: [])
        } catch {
            throw AlertManagerError.audioSessionConfigurationFailed(error.localizedDescription)
        }
        registerAudioSessionObservers()
    }

    private func loadAudioAssets() throws {
        chimeBuffer = try Self.loadPCMBuffer(
            resource: configuration.chimeResourceName,
            extension: configuration.chimeResourceExtension
        )
        sirenBuffer = try Self.loadPCMBuffer(
            resource: configuration.sirenResourceName,
            extension: configuration.sirenResourceExtension
        )
    }

    /// Decodes a bundled audio asset fully into memory once, up front, so
    /// alert playback at detection time never touches disk I/O or a codec.
    private nonisolated static func loadPCMBuffer(resource: String, extension ext: String) throws -> AVAudioPCMBuffer {
        guard let url = Bundle.main.url(forResource: resource, withExtension: ext) else {
            throw AlertManagerError.audioAssetNotFound("\(resource).\(ext)")
        }
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AlertManagerError.audioAssetLoadFailed(error.localizedDescription)
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw AlertManagerError.audioAssetLoadFailed("Could not allocate PCM buffer for \(resource).\(ext).")
        }
        do {
            try file.read(into: buffer)
        } catch {
            throw AlertManagerError.audioAssetLoadFailed(error.localizedDescription)
        }
        return buffer
    }

    private func prepareAudioEngine() throws {
        audioEngine.attach(playerNode)
        guard let format = chimeBuffer?.format ?? sirenBuffer?.format else {
            throw AlertManagerError.audioAssetLoadFailed("No decoded audio format available to configure the mixer.")
        }
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            throw AlertManagerError.audioEngineStartFailed(error.localizedDescription)
        }
    }

    private func prepareHapticEngine() throws {
        do {
            let engine = try CHHapticEngine()
            engine.isAutoShutdownEnabled = false // stay resident; we manage lifecycle explicitly for low latency
            engine.resetHandler = { [weak self] in
                guard let self else { return }
                Task { await self.handleHapticEngineReset() }
            }
            engine.stoppedHandler = { [weak self] reason in
                guard let self else { return }
                Task { await self.handleHapticEngineStopped(reason: reason) }
            }
            try engine.start()
            hapticEngine = engine

            drowsyPatternPlayer = try Self.makePulsePlayer(
                engine: engine,
                intensity: configuration.drowsyHapticIntensity,
                sharpness: configuration.drowsyHapticSharpness,
                pulseCount: 1,
                spacing: 0
            )
            microsleepPatternPlayer = try Self.makeLoopingPulsePlayer(
                engine: engine,
                intensity: configuration.microsleepHapticIntensity,
                sharpness: configuration.microsleepHapticSharpness,
                spacing: configuration.microsleepPulseSpacing
            )
        } catch {
            throw AlertManagerError.hapticEngineStartFailed(error.localizedDescription)
        }
    }

    private nonisolated static func makePulsePlayer(
        engine: CHHapticEngine,
        intensity: Float,
        sharpness: Float,
        pulseCount: Int,
        spacing: TimeInterval
    ) throws -> CHHapticPatternPlayer {
        let events = (0..<max(pulseCount, 1)).map { index in
            CHHapticEvent(
                eventType: .hapticTransient,
                parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: sharpness)
                ],
                relativeTime: Double(index) * spacing
            )
        }
        let pattern = try CHHapticPattern(events: events, parameters: [])
        return try engine.makePlayer(with: pattern)
    }

    /// Builds a short repeating unit and lets `CHHapticAdvancedPatternPlayer`
    /// loop it indefinitely, rather than constructing an arbitrarily long
    /// pattern up front for an alarm whose duration isn't known ahead of time.
    private nonisolated static func makeLoopingPulsePlayer(
        engine: CHHapticEngine,
        intensity: Float,
        sharpness: Float,
        spacing: TimeInterval
    ) throws -> CHHapticAdvancedPatternPlayer {
        let pulseCount = 3
        let events = (0..<pulseCount).map { index in
            CHHapticEvent(
                eventType: .hapticTransient,
                parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: sharpness)
                ],
                relativeTime: Double(index) * spacing
            )
        }
        let pattern = try CHHapticPattern(events: events, parameters: [])
        let player = try engine.makeAdvancedPlayer(with: pattern)
        player.loopEnabled = true
        player.loopEnd = Double(pulseCount) * spacing
        return player
    }

    // MARK: - Haptic engine resilience

    private func handleHapticEngineReset() {
        logger.warning("Haptic engine reset by the system; restarting and rebuilding pattern players.")
        guard let engine = hapticEngine else { return }
        do {
            try engine.start()
            drowsyPatternPlayer = try Self.makePulsePlayer(
                engine: engine,
                intensity: configuration.drowsyHapticIntensity,
                sharpness: configuration.drowsyHapticSharpness,
                pulseCount: 1,
                spacing: 0
            )
            microsleepPatternPlayer = try Self.makeLoopingPulsePlayer(
                engine: engine,
                intensity: configuration.microsleepHapticIntensity,
                sharpness: configuration.microsleepHapticSharpness,
                spacing: configuration.microsleepPulseSpacing
            )
            // A microsleep alarm is safety-critical: if the engine reset
            // mid-alarm, resume the haptic component immediately.
            if currentState == .microsleep {
                try? microsleepPatternPlayer?.start(atTime: CHHapticTimeImmediate)
            }
        } catch {
            logger.error("Failed to restart haptic engine after reset: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func handleHapticEngineStopped(reason: CHHapticEngine.StoppedReason) {
        logger.warning("Haptic engine stopped: \(String(describing: reason), privacy: .public)")
    }

    // MARK: - Audio session resilience

    private func registerAudioSessionObservers() {
        guard interruptionObserver == nil else { return }
        let center = NotificationCenter.default

        interruptionObserver = center.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance(),
            queue: nil
        ) { [weak self] notification in
            guard let self else { return }
            Task { await self.handleAudioInterruption(notification) }
        }

        routeChangeObserver = center.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: AVAudioSession.sharedInstance(),
            queue: nil
        ) { [weak self] notification in
            guard let self else { return }
            Task { await self.handleRouteChange(notification) }
        }
    }

    private func handleAudioInterruption(_ notification: Notification) {
        guard let info = notification.userInfo,
              let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else { return }

        switch type {
        case .began:
            logger.warning("Audio session interrupted (e.g. phone call); audio alert paused by the system.")

        case .ended:
            let optionsValue = info[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
            guard AVAudioSession.InterruptionOptions(rawValue: optionsValue).contains(.shouldResume) else { return }
            resumeAudioAfterInterruption()

        @unknown default:
            break
        }
    }

    private func resumeAudioAfterInterruption() {
        do {
            try AVAudioSession.sharedInstance().setActive(true, options: [])
            if !audioEngine.isRunning {
                try audioEngine.start()
            }
            // Safety-critical: resume the alarm immediately rather than
            // waiting for the next fatigue assessment to re-trigger it.
            if currentState == .microsleep {
                playLoopingSiren()
            }
        } catch {
            logger.error("Failed to resume audio session after interruption: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func handleRouteChange(_ notification: Notification) {
        guard let info = notification.userInfo,
              let reasonValue = info[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else { return }

        // e.g. headphones/CarPlay disconnected mid-alarm. The system may
        // pause playback on route loss; explicitly re-assert the alarm
        // through whatever route is now active.
        if reason == .oldDeviceUnavailable, currentState == .microsleep {
            playLoopingSiren()
        }
    }

    // MARK: - Public API

    /// Primary entry point. Call with each new `FatigueAssessment.state`
    /// from `EyeAspectCalculator`. Only state *transitions* trigger new
    /// alert behavior — repeated identical states are no-ops — except the
    /// microsleep alarm, which is already looping continuously and needs no
    /// re-triggering while sustained.
    func handle(state: DriverState) {
        guard isConfigured else {
            logger.error("handle(state:) called before configure() succeeded; ignoring.")
            return
        }
        guard state != currentState else { return }

        let previousState = currentState
        currentState = state
        drowsyReminderTask?.cancel()
        drowsyReminderTask = nil

        switch state {
        case .alert:
            stopAllAlerts()
        case .drowsy:
            stopSiren() // in case we're de-escalating down from microsleep
            triggerDrowsyAlert()
            scheduleDrowsyReminders()
        case .microsleep:
            triggerMicrosleepAlarm()
        }

        logger.info("Alert escalation: \(String(describing: previousState), privacy: .public) -> \(String(describing: state), privacy: .public)")
    }

    /// Stops everything and tears down the audio/haptic stack. Call when
    /// monitoring ends (trip finished, feature disabled, app terminating).
    func shutdown() {
        drowsyReminderTask?.cancel()
        drowsyReminderTask = nil
        stopAllAlerts()

        audioEngine.stop()
        hapticEngine?.stop()

        if let interruptionObserver {
            NotificationCenter.default.removeObserver(interruptionObserver)
        }
        if let routeChangeObserver {
            NotificationCenter.default.removeObserver(routeChangeObserver)
        }
        interruptionObserver = nil
        routeChangeObserver = nil

        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        isConfigured = false
        currentState = .alert
        logger.info("AlertManager shut down.")
    }

    // MARK: - Alert actions

    private func triggerDrowsyAlert() {
        playOneShot(buffer: chimeBuffer)
        if let player = drowsyPatternPlayer {
            try? player.start(atTime: CHHapticTimeImmediate)
        } else if !hapticsSupported {
            fireFallbackImpact(style: .medium)
        }
    }

    private func triggerMicrosleepAlarm() {
        playLoopingSiren()
        if let player = microsleepPatternPlayer {
            try? player.start(atTime: CHHapticTimeImmediate)
        } else if !hapticsSupported {
            fireFallbackNotification(type: .warning)
        }
    }

    private func scheduleDrowsyReminders() {
        let interval = configuration.drowsyReminderInterval
        guard interval > 0 else { return }
        drowsyReminderTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                guard !Task.isCancelled else { return }
                await self.reissueDrowsyReminderIfStillDrowsy()
            }
        }
    }

    private func reissueDrowsyReminderIfStillDrowsy() {
        guard currentState == .drowsy else { return }
        triggerDrowsyAlert()
    }

    private func stopAllAlerts() {
        stopSiren()
        if let drowsyPatternPlayer {
            try? drowsyPatternPlayer.stop(atTime: CHHapticTimeImmediate)
        }
        if let microsleepPatternPlayer {
            try? microsleepPatternPlayer.stop(atTime: CHHapticTimeImmediate)
        }
    }

    private func stopSiren() {
        playerNode.stop()
    }

    private func playOneShot(buffer: AVAudioPCMBuffer?) {
        guard let buffer else {
            logger.warning("Chime buffer unavailable; drowsy alert is haptics-only.")
            return
        }
        ensureAudioEngineRunning()
        playerNode.stop()
        playerNode.scheduleBuffer(buffer, at: nil, options: [.interrupts], completionHandler: nil)
        playerNode.play()
    }

    private func playLoopingSiren() {
        guard let buffer = sirenBuffer else {
            logger.warning("Siren buffer unavailable; microsleep alarm is haptics-only.")
            return
        }
        ensureAudioEngineRunning()
        playerNode.stop()
        playerNode.scheduleBuffer(buffer, at: nil, options: [.loops, .interrupts], completionHandler: nil)
        playerNode.play()
    }

    private func ensureAudioEngineRunning() {
        guard !audioEngine.isRunning else { return }
        do {
            try audioEngine.start()
        } catch {
            logger.error("Failed to restart audio engine on demand: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - UIFeedbackGenerator fallback (devices without CoreHaptics)

    private nonisolated func fireFallbackImpact(style: UIImpactFeedbackGenerator.FeedbackStyle) {
        Task { @MainActor in
            let generator = UIImpactFeedbackGenerator(style: style)
            generator.prepare()
            generator.impactOccurred()
        }
    }

    private nonisolated func fireFallbackNotification(type: UINotificationFeedbackGenerator.FeedbackType) {
        Task { @MainActor in
            let generator = UINotificationFeedbackGenerator()
            generator.prepare()
            generator.notificationOccurred(type)
        }
    }
}