//
//  CameraManager.swift
//  Astrateq Vision
//
//  Privacy-first, on-device driver awareness camera pipeline.
//  Streams CVPixelBuffer frames from the front camera to a MediaPipe/ANE
//  inference pipeline via AsyncStream, with zero video persistence.
//
//  Requires: iOS 17+, Swift 6 (strict concurrency)
//

import AVFoundation
import CoreVideo
import os.log

// MARK: - Errors

enum CameraError: Error, LocalizedError, Sendable {
    case permissionDenied
    case permissionRestricted
    case noFrontCamera
    case cannotAddInput
    case cannotAddOutput
    case configurationFailed(String)
    case sessionInterrupted(reason: String)
    case runtimeError(String)

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Camera access was denied. Enable it in Settings to use driver monitoring."
        case .permissionRestricted:
            return "Camera access is restricted on this device."
        case .noFrontCamera:
            return "No front-facing camera is available on this device."
        case .cannotAddInput:
            return "Failed to add camera input to the capture session."
        case .cannotAddOutput:
            return "Failed to add video output to the capture session."
        case .configurationFailed(let detail):
            return "Camera configuration failed: \(detail)"
        case .sessionInterrupted(let reason):
            return "Capture session interrupted: \(reason)"
        case .runtimeError(let detail):
            return "Camera runtime error: \(detail)"
        }
    }
}

// MARK: - Camera State

enum CameraSessionState: Sendable, Equatable {
    case idle
    case configuring
    case running
    case interrupted(reason: String)
    case stopped
    case failed(String)
}

// MARK: - CameraManager

/// Actor-isolated manager owning the AVCaptureSession lifecycle and frame
/// distribution. All mutable capture state is confined to this actor;
/// AVFoundation delegate callbacks arrive on a private serial queue and are
/// funneled back in via a non-isolated, Sendable-safe bridge.
actor CameraManager {

    // MARK: Public stream

    /// Downstream consumers (inference pipeline) iterate this stream.
    /// Buffers are retained only for the duration of one iteration —
    /// consumers must not hold references beyond their processing scope.
    private(set) var frameStream: AsyncStream<CVPixelBuffer>!
    private var frameContinuation: AsyncStream<CVPixelBuffer>.Continuation!

    private(set) var state: CameraSessionState = .idle

    // MARK: AVFoundation objects

    private let session = AVCaptureSession()
    private var videoInput: AVCaptureDeviceInput?
    private let videoOutput = AVCaptureVideoDataOutput()
    private var pixelBufferPool: CVPixelBufferPool?

    /// Dedicated serial queue for session config + delegate callbacks.
    /// Never touch this from the main thread.
    private let sessionQueue = DispatchQueue(
        label: "com.astrateq.vision.camera.session",
        qos: .userInitiated
    )

    /// Bridges AVCaptureVideoDataOutputSampleBufferDelegate (non-actor,
    /// synchronous, arbitrary queue) into the actor's async world.
    private var delegateBridge: FrameDelegateBridge?

    private let logger = Logger(subsystem: "com.astrateq.vision", category: "CameraManager")

    // Frame-drop / thermal telemetry
    private(set) var droppedFrameCount: Int = 0
    private(set) var deliveredFrameCount: Int = 0

    // Desired throughput; MediaPipe Face Landmarker typically targets 30fps,
    // but we allow up to 60 for higher-fidelity EAR sampling on capable devices.
    private let targetFrameRate: Double

    init(targetFrameRate: Double = 30.0) {
        self.targetFrameRate = targetFrameRate
        var continuation: AsyncStream<CVPixelBuffer>.Continuation!
        let stream = AsyncStream<CVPixelBuffer> { cont in
            continuation = cont
        }
        self.frameStream = stream
        self.frameContinuation = continuation
    }

    deinit {
        frameContinuation?.finish()
    }

    // MARK: - Public API

    /// Requests permission (if needed), configures the session, and starts
    /// streaming. Safe to call once at app/feature launch.
    func start() async throws {
        try await requestPermissionIfNeeded()
        try await configureSessionIfNeeded()
        await startRunning()
    }

    /// Stops the session and finishes the frame stream. Call on view
    /// disappearance / app backgrounding to guarantee no camera activity
    /// continues off-screen.
    func stop() async {
        guard session.isRunning else { return }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            sessionQueue.async { [weak self] in
                self?.session.stopRunning()
                continuation.resume()
            }
        }
        state = .stopped
        logger.info("Capture session stopped.")
    }

    /// Tears down the stream permanently (e.g. on deinit of the owning feature).
    func shutdown() {
        frameContinuation.finish()
        if session.isRunning {
            session.stopRunning()
        }
    }

    /// Current dropped/delivered counters, useful for thermal/perf HUDs.
    func frameStatistics() -> (delivered: Int, dropped: Int) {
        (deliveredFrameCount, droppedFrameCount)
    }

    // MARK: - Permissions

    private func requestPermissionIfNeeded() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            if !granted {
                state = .failed(CameraError.permissionDenied.localizedDescription)
                throw CameraError.permissionDenied
            }
        case .denied:
            state = .failed(CameraError.permissionDenied.localizedDescription)
            throw CameraError.permissionDenied
        case .restricted:
            state = .failed(CameraError.permissionRestricted.localizedDescription)
            throw CameraError.permissionRestricted
        @unknown default:
            throw CameraError.permissionDenied
        }
    }

    // MARK: - Configuration

    private var isConfigured = false

    private func configureSessionIfNeeded() async throws {
        guard !isConfigured else { return }
        state = .configuring

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async { [weak self] in
                guard let self else { return }
                do {
                    try self.configureSessionSync()
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }

        isConfigured = true
    }

    /// Runs entirely on `sessionQueue`. Marked `nonisolated` because it must
    /// execute synchronously off-actor to satisfy AVCaptureSession's
    /// begin/commitConfiguration contract without hopping back and forth.
    nonisolated private func configureSessionSync() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .high

        // --- Input: front camera ---
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            throw CameraError.noFrontCamera
        }

        do {
            try device.lockForConfiguration()
            if let bestRange = device.formats
                .flatMap({ $0.videoSupportedFrameRateRanges })
                .filter({ $0.maxFrameRate >= targetFrameRate })
                .min(by: { $0.maxFrameRate < $1.maxFrameRate }) {
                device.activeVideoMinFrameDuration = CMTime(
                    value: 1, timescale: CMTimeScale(targetFrameRate)
                )
                device.activeVideoMaxFrameDuration = CMTime(
                    value: 1, timescale: CMTimeScale(min(targetFrameRate, bestRange.maxFrameRate))
                )
            }
            if device.isLowLightBoostSupported {
                device.automaticallyEnablesLowLightBoostWhenAvailable = true
            }
            device.unlockForConfiguration()
        } catch {
            throw CameraError.configurationFailed("Unable to lock device for configuration: \(error.localizedDescription)")
        }

        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            throw CameraError.configurationFailed("Unable to create device input: \(error.localizedDescription)")
        }

        guard session.canAddInput(input) else {
            throw CameraError.cannotAddInput
        }
        session.addInput(input)
        self.videoInput = input

        // --- Output: video data (BGRA for MediaPipe / Vision compatibility) ---
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true // critical: prevents backlog under thermal pressure

        let bridge = FrameDelegateBridge(manager: self)
        videoOutput.setSampleBufferDelegate(bridge, queue: sessionQueue)
        self.delegateBridge = bridge

        guard session.canAddOutput(videoOutput) else {
            throw CameraError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        // Mirror + orientation for the front camera preview/inference consistency.
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90 // portrait
            }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = true
            }
        }

        // Own pixel buffer pool for recycled output buffers (used when we
        // need to hand MediaPipe a buffer decoupled from AVFoundation's
        // internal pool lifetime, e.g. after color conversion/copy).
        let poolAttributes: [String: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: 3
        ]
        let pixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 640,
            kCVPixelBufferHeightKey as String: 480,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        var pool: CVPixelBufferPool?
        let status = CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            poolAttributes as CFDictionary,
            pixelBufferAttributes as CFDictionary,
            &pool
        )
        if status == kCVReturnSuccess {
            self.pixelBufferPool = pool
        } else {
            logger.warning("Failed to create CVPixelBufferPool (status \(status)); falling back to passthrough buffers.")
        }
    }

    // MARK: - Session lifecycle

    private func startRunning() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            sessionQueue.async { [weak self] in
                guard let self else {
                    continuation.resume()
                    return
                }
                if !self.session.isRunning {
                    self.session.startRunning()
                }
                continuation.resume()
            }
        }
        state = .running
        registerInterruptionObservers()
        logger.info("Capture session running at target \(self.targetFrameRate, privacy: .public) fps.")
    }

    // MARK: - Interruption handling

    private var notificationObservers: [NSObjectProtocol] = []

    private func registerInterruptionObservers() {
        guard notificationObservers.isEmpty else { return }
        let center = NotificationCenter.default

        let interruption = center.addObserver(
            forName: .AVCaptureSessionWasInterrupted,
            object: session,
            queue: nil
        ) { [weak self] notification in
            let reasonValue = (notification.userInfo?[AVCaptureSessionInterruptionReasonKey] as? NSNumber)?.intValue
            let reason = CameraManager.describeInterruptionReason(reasonValue)
            Task { await self?.handleInterruption(reason: reason) }
        }

        let ended = center.addObserver(
            forName: .AVCaptureSessionInterruptionEnded,
            object: session,
            queue: nil
        ) { [weak self] _ in
            Task { await self?.handleInterruptionEnded() }
        }

        let runtimeError = center.addObserver(
            forName: .AVCaptureSessionRuntimeError,
            object: session,
            queue: nil
        ) { [weak self] notification in
            let error = notification.userInfo?[AVCaptureSessionErrorKey] as? AVError
            Task { await self?.handleRuntimeError(error) }
        }

        notificationObservers = [interruption, ended, runtimeError]
    }

    private nonisolated static func describeInterruptionReason(_ raw: Int?) -> String {
        guard let raw, let reason = AVCaptureSession.InterruptionReason(rawValue: raw) else {
            return "unknown"
        }
        switch reason {
        case .videoDeviceNotAvailableInBackground: return "camera unavailable in background"
        case .audioDeviceInUseByAnotherClient: return "audio device in use elsewhere"
        case .videoDeviceInUseByAnotherClient: return "camera in use by another app"
        case .videoDeviceNotAvailableWithMultipleForegroundApps: return "unavailable in multi-app mode"
        case .videoDeviceNotAvailableDueToSystemPressure: return "system thermal/resource pressure"
        @unknown default: return "unrecognized reason"
        }
    }

    private func handleInterruption(reason: String) {
        state = .interrupted(reason: reason)
        logger.warning("Session interrupted: \(reason, privacy: .public)")
    }

    private func handleInterruptionEnded() {
        guard case .interrupted = state else { return }
        state = .running
        logger.info("Session interruption ended; resumed.")
    }

    private func handleRuntimeError(_ error: AVError?) {
        let message = error?.localizedDescription ?? "unknown runtime error"
        state = .failed(message)
        logger.error("Runtime error: \(message, privacy: .public)")

        // Attempt one automatic restart off the main actor.
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
                Task { await self.markRecovered() }
            }
        }
    }

    private func markRecovered() {
        state = .running
        logger.info("Session auto-recovered after runtime error.")
    }

    // MARK: - Frame ingestion (called by delegate bridge)

    /// Invoked by `FrameDelegateBridge` for every captured sample buffer.
    /// Copies the buffer into our pool (when available) to fully decouple
    /// lifetime from AVFoundation's internal buffer pool, then yields it.
    fileprivate func ingest(sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            droppedFrameCount += 1
            return
        }

        let outputBuffer: CVPixelBuffer
        if let recycled = recycledBuffer(copying: pixelBuffer) {
            outputBuffer = recycled
        } else {
            // Fallback: pass the original through. Caller must not retain
            // it past processing, since AVFoundation may reuse it.
            outputBuffer = pixelBuffer
        }

        switch frameContinuation.yield(outputBuffer) {
        case .enqueued:
            deliveredFrameCount += 1
        case .dropped, .terminated:
            droppedFrameCount += 1
        @unknown default:
            droppedFrameCount += 1
        }
    }

    /// Copies `source` into a buffer drawn from our recycling pool so the
    /// inference pipeline can hold it beyond the delegate callback's
    /// lifetime without pinning AVFoundation's own pool.
    private func recycledBuffer(copying source: CVPixelBuffer) -> CVPixelBuffer? {
        guard let pool = pixelBufferPool else { return nil }

        var maybeBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &maybeBuffer)
        guard status == kCVReturnSuccess, let destination = maybeBuffer else {
            logger.warning("Pixel buffer pool exhausted (status \(status)); dropping frame to avoid backlog.")
            return nil
        }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }

        let height = min(CVPixelBufferGetHeight(source), CVPixelBufferGetHeight(destination))
        let srcRowBytes = CVPixelBufferGetBytesPerRow(source)
        let dstRowBytes = CVPixelBufferGetBytesPerRow(destination)
        let copyBytes = min(srcRowBytes, dstRowBytes)

        guard let srcBase = CVPixelBufferGetBaseAddress(source),
              let dstBase = CVPixelBufferGetBaseAddress(destination) else {
            return nil
        }

        for row in 0..<height {
            let srcRow = srcBase.advanced(by: row * srcRowBytes)
            let dstRow = dstBase.advanced(by: row * dstRowBytes)
            memcpy(dstRow, srcRow, copyBytes)
        }

        return destination
    }
}

// MARK: - Delegate Bridge

/// AVCaptureVideoDataOutputSampleBufferDelegate cannot be an `actor` (it
/// must be a plain NSObject-conforming class for Objective-C interop), so
/// this small `Sendable` bridge receives callbacks on `sessionQueue` and
/// forwards them into the actor via a `Task`. Kept intentionally minimal —
/// no state beyond a weak reference to avoid retain cycles with the session.
private final class FrameDelegateBridge: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private weak var manager: CameraManager?

    init(manager: CameraManager) {
        self.manager = manager
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // `alwaysDiscardsLateVideoFrames = true` on the output means this
        // delegate is never called with a backlog; each call is the latest
        // frame. We still hop through the actor to keep all mutable state
        // (counters, continuation) single-threaded.
        guard let manager else { return }
        Task { await manager.ingest(sampleBuffer: sampleBuffer) }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // AVFoundation dropped the frame before delivery (e.g. downstream
        // congestion). Nothing to ingest, but useful for future telemetry
        // hooks if you want to log drop reasons via CMSampleBufferGetAttachment.
    }
}