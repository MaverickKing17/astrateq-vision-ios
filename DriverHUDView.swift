//
//  DriverHUDView.swift
//  Astrateq Vision
//
//  High-contrast, night-driving-optimized heads-up display. Wires
//  CameraManager -> FaceLandmarkerService -> EyeAspectCalculator ->
//  AlertManager into a single @Observable view model and renders live
//  telemetry with minimal, non-distracting chrome.
//
//  Requires: iOS 17+, Swift 6 (strict concurrency)
//
//  ─────────────────────────────────────────────────────────────────────
//  INTEGRATION NOTE — CameraManager access
//  This view renders a live AVCaptureVideoPreviewLayer bound directly to
//  CameraManager's capture session. That requires the `session` property
//  on `CameraManager` to be visible outside the actor as a stable,
//  non-isolated reference. If you're using the `CameraManager.swift`
//  delivered earlier in this project, change its declaration from:
//
//      private let session = AVCaptureSession()
//  to:
//      nonisolated(unsafe) let session = AVCaptureSession()
//
//  AVCaptureSession is safe to hand to a preview layer from any thread for
//  this narrow purpose (Apple's own preview-layer sample code does the
//  same); all *configuration* of the session still funnels exclusively
//  through the actor's `sessionQueue` as before. `nonisolated(unsafe)`
//  documents this as a deliberate, scoped exception rather than an
//  oversight. No other file needs to change.
//  ─────────────────────────────────────────────────────────────────────
//

import SwiftUI
import AVFoundation
import CoreVideo
import QuartzCore

// MARK: - CameraManager bridging (UI-only, read-only)

extension CameraManager {
    /// Stable reference to the underlying capture session, exposed solely
    /// so SwiftUI can bind an `AVCaptureVideoPreviewLayer` to it. Never
    /// mutate the session through this reference — configuration remains
    /// the actor's exclusive responsibility.
    nonisolated var previewSession: AVCaptureSession { session }
}

// MARK: - Camera Preview (UIViewRepresentable)

private struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewLayerContainerView {
        let view = PreviewLayerContainerView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: PreviewLayerContainerView, context: Context) {
        // Session reference is stable for the view's lifetime; nothing to update.
    }

    final class PreviewLayerContainerView: UIView {
        override static var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            // swiftlint:disable:next force_cast
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}

// MARK: - Lightweight smoothing (O(1), no allocation)

/// Exponential moving average used to smooth FPS and latency readouts so
/// the HUD numbers don't jitter every single frame — cheap enough to run
/// on every callback without measurable overhead.
private struct ExponentialMovingAverage {
    private var value: Double?
    let smoothing: Double

    init(smoothing: Double = 0.2) {
        self.smoothing = smoothing
    }

    mutating func push(_ sample: Double) -> Double {
        guard let current = value else {
            value = sample
            return sample
        }
        let updated = current + smoothing * (sample - current)
        value = updated
        return updated
    }
}

// MARK: - Driver HUD View Model

/// Owns the async pipelines connecting the four vision/alert actors and
/// republishes their output as plain, main-actor-isolated properties via
/// the `@Observable` macro, so `DriverHUDView` gets automatic, granular
/// SwiftUI invalidation with no manual Combine wiring.
@MainActor
@Observable
final class DriverHUDViewModel {

    // MARK: Published telemetry (read-only outside this type)

    private(set) var driverState: DriverState = .alert
    private(set) var currentEAR: Double = 0
    private(set) var perclos30: Double = 0
    private(set) var perclos60: Double = 0
    private(set) var framesPerSecond: Double = 0
    private(set) var inferenceLatencyMs: Double = 0
    private(set) var isFaceVisible: Bool = false
    private(set) var startupError: String?

    // MARK: Dependencies

    let cameraManager: CameraManager
    private let faceLandmarkerService: FaceLandmarkerService
    private let eyeAspectCalculator: EyeAspectCalculator
    private let alertManager: AlertManager

    // MARK: Internal pipeline state

    private var captureTask: Task<Void, Never>?
    private var detectionTask: Task<Void, Never>?
    private var fpsAverage = ExponentialMovingAverage(smoothing: 0.2)
    private var latencyAverage = ExponentialMovingAverage(smoothing: 0.3)
    private var lastFrameTimestamp: TimeInterval?

    init(
        cameraManager: CameraManager,
        faceLandmarkerService: FaceLandmarkerService,
        eyeAspectCalculator: EyeAspectCalculator,
        alertManager: AlertManager
    ) {
        self.cameraManager = cameraManager
        self.faceLandmarkerService = faceLandmarkerService
        self.eyeAspectCalculator = eyeAspectCalculator
        self.alertManager = alertManager
    }

    // MARK: - Lifecycle

    func start() async {
        do {
            try await faceLandmarkerService.start()
            try await alertManager.configure()
            try await cameraManager.start()
        } catch {
            startupError = error.localizedDescription
            return
        }

        detectionTask = Task { [weak self] in
            guard let self else { return }
            for await output in await self.faceLandmarkerService.resultStream {
                await self.handleLandmarkOutput(output)
            }
        }

        captureTask = Task { [weak self] in
            guard let self else { return }
            for await pixelBuffer in await self.cameraManager.frameStream {
                await self.submitFrame(pixelBuffer)
            }
        }
    }

    func stop() async {
        captureTask?.cancel()
        detectionTask?.cancel()
        captureTask = nil
        detectionTask = nil
        await cameraManager.stop()
        await faceLandmarkerService.shutdown()
        await alertManager.shutdown()
    }

    // MARK: - Pipeline steps

    private func submitFrame(_ pixelBuffer: CVPixelBuffer) async {
        recordFrameArrival()
        let timestampMs = Int(CACurrentMediaTime() * 1000)
        do {
            try await faceLandmarkerService.detectAsync(pixelBuffer: pixelBuffer, timestampMs: timestampMs)
        } catch {
            // Transient per-frame detection failures shouldn't halt the
            // stream; the next frame gets another chance.
        }
    }

    private func handleLandmarkOutput(_ output: FaceLandmarkerOutput) async {
        let nowMs = Int(CACurrentMediaTime() * 1000)
        inferenceLatencyMs = latencyAverage.push(Double(max(nowMs - output.timestampMs, 0)))

        guard let left = output.leftEye, let right = output.rightEye else {
            isFaceVisible = false
            return
        }
        isFaceVisible = true

        let assessment = await eyeAspectCalculator.ingest(
            leftEye: left,
            rightEye: right,
            timestamp: Double(output.timestampMs) / 1000
        )

        currentEAR = assessment.averageEAR
        perclos30 = assessment.perclos30
        perclos60 = assessment.perclos60

        if assessment.state != driverState {
            withAnimation(.easeInOut(duration: 0.35)) {
                driverState = assessment.state
            }
        }

        await alertManager.handle(state: assessment.state)
    }

    private func recordFrameArrival() {
        let now = CACurrentMediaTime()
        defer { lastFrameTimestamp = now }
        guard let last = lastFrameTimestamp else { return }
        let dt = now - last
        guard dt > 0 else { return }
        framesPerSecond = fpsAverage.push(1.0 / dt)
    }
}

// MARK: - Driver HUD View

struct DriverHUDView: View {
    @State private var viewModel: DriverHUDViewModel
    @State private var microsleepPulseActive = false
    @Environment(\.scenePhase) private var scenePhase

    init(
        cameraManager: CameraManager,
        faceLandmarkerService: FaceLandmarkerService,
        eyeAspectCalculator: EyeAspectCalculator,
        alertManager: AlertManager
    ) {
        _viewModel = State(
            initialValue: DriverHUDViewModel(
                cameraManager: cameraManager,
                faceLandmarkerService: faceLandmarkerService,
                eyeAspectCalculator: eyeAspectCalculator,
                alertManager: alertManager
            )
        )
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            cameraBackdrop

            VStack {
                driverStateBanner
                    .padding(.top, 24)

                Spacer()

                if let startupError = viewModel.startupError {
                    startupErrorCard(startupError)
                }

                telemetryPanel

                diagnosticsFooter
                    .padding(.bottom, 12)
            }
            .padding(.horizontal, 20)

            microsleepOverlay
        }
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .task {
            UIApplication.shared.isIdleTimerDisabled = true
            await viewModel.start()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            Task { await viewModel.stop() }
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background {
                Task { await viewModel.stop() }
            }
        }
        .onChange(of: viewModel.driverState) { _, newState in
            microsleepPulseActive = (newState == .microsleep)
        }
    }

    // MARK: - Camera backdrop

    /// Low-opacity live feed for ambient context only — not relied upon for
    /// any vision accuracy, which happens entirely on the raw buffer stream
    /// independent of what's rendered here.
    private var cameraBackdrop: some View {
        CameraPreviewView(session: viewModel.cameraManager.previewSession)
            .opacity(0.22)
            .ignoresSafeArea()
            .overlay(Color.black.opacity(0.35).ignoresSafeArea())
    }

    // MARK: - Driver state banner

    private var driverStateBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: bannerSymbolName)
                .font(.system(size: 26, weight: .bold))
            Text(bannerTitle)
                .font(.system(.title3, design: .rounded, weight: .heavy))
                .tracking(0.5)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 26)
        .padding(.vertical, 14)
        .background(
            Capsule()
                .fill(bannerColor.gradient)
                .shadow(
                    color: bannerColor.opacity(viewModel.driverState == .microsleep ? 0.8 : 0.35),
                    radius: viewModel.driverState == .microsleep ? 20 : 8
                )
        )
        .animation(.easeInOut(duration: 0.35), value: viewModel.driverState)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Driver state: \(bannerTitle)")
    }

    private var bannerColor: Color {
        switch viewModel.driverState {
        case .alert: return Color(red: 0.20, green: 0.50, blue: 0.32)     // muted green — low luminance for night driving
        case .drowsy: return Color(red: 0.82, green: 0.55, blue: 0.10)    // warning amber
        case .microsleep: return Color(red: 0.86, green: 0.12, blue: 0.12) // alarm red
        }
    }

    private var bannerSymbolName: String {
        switch viewModel.driverState {
        case .alert: return "checkmark.circle.fill"
        case .drowsy: return "exclamationmark.triangle.fill"
        case .microsleep: return "exclamationmark.octagon.fill"
        }
    }

    private var bannerTitle: String {
        switch viewModel.driverState {
        case .alert: return "ALERT"
        case .drowsy: return "DROWSY — TAKE A BREAK"
        case .microsleep: return "WAKE UP"
        }
    }

    // MARK: - Microsleep full-screen pulse

    /// Continuous pulsing red overlay while in `.microsleep`. Toggling
    /// `microsleepPulseActive` once (on state entry) is sufficient — the
    /// `repeatForever` animation curve keeps the presentation layer
    /// oscillating indefinitely until the value flips back.
    private var microsleepOverlay: some View {
        Color.red
            .opacity(viewModel.driverState == .microsleep ? (microsleepPulseActive ? 0.32 : 0.06) : 0)
            .ignoresSafeArea()
            .allowsHitTesting(false)
            .animation(
                viewModel.driverState == .microsleep
                    ? .easeInOut(duration: 0.5).repeatForever(autoreverses: true)
                    : .easeInOut(duration: 0.3),
                value: microsleepPulseActive
            )
    }

    // MARK: - Telemetry panel

    private var telemetryPanel: some View {
        HStack(spacing: 24) {
            earGauge
            perclosGauge
        }
        .padding(20)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
        )
    }

    private var earGauge: some View {
        Gauge(value: viewModel.currentEAR, in: 0...0.45) {
            Text("EAR")
        } currentValueLabel: {
            Text(String(format: "%.2f", viewModel.currentEAR))
                .font(.system(.body, design: .rounded, weight: .bold))
        } minimumValueLabel: {
            Text("0")
        } maximumValueLabel: {
            Text(".45")
        }
        .gaugeStyle(.accessoryCircular)
        .tint(Gradient(colors: [.red, .orange, .green]))
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity)
        .accessibilityLabel("Eye aspect ratio")
        .accessibilityValue(String(format: "%.2f", viewModel.currentEAR))
    }

    private var perclosGauge: some View {
        Gauge(value: viewModel.perclos30, in: 0...1) {
            Text("PERCLOS")
        } currentValueLabel: {
            Text("\(Int((viewModel.perclos30 * 100).rounded()))%")
                .font(.system(.body, design: .rounded, weight: .bold))
        }
        .gaugeStyle(.accessoryCircularCapacity)
        .tint(perclosGaugeTint)
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity)
        .accessibilityLabel("30 second eye closure percentage")
        .accessibilityValue("\(Int((viewModel.perclos30 * 100).rounded())) percent")
    }

    private var perclosGaugeTint: Color {
        switch viewModel.perclos30 {
        case ..<0.15: return .green
        case 0.15..<0.30: return .orange
        default: return .red
        }
    }

    // MARK: - Diagnostics footer

    private var diagnosticsFooter: some View {
        HStack(spacing: 18) {
            Label("\(Int(viewModel.framesPerSecond.rounded())) FPS", systemImage: "gauge.with.dots.needle.67percent")
            Label("\(Int(viewModel.inferenceLatencyMs.rounded())) ms", systemImage: "timer")
            if !viewModel.isFaceVisible {
                Label("No face detected", systemImage: "person.crop.circle.badge.exclamationmark")
                    .foregroundStyle(.yellow)
            }
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.white.opacity(0.5))
        .accessibilityHidden(true) // diagnostic-only; not meaningful to VoiceOver drivers
    }

    // MARK: - Startup error state

    private func startupErrorCard(_ message: String) -> some View {
        VStack(spacing: 8) {
            Label("Monitoring unavailable", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundStyle(.yellow)
            Text(message)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .padding(.horizontal, 8)
    }
}

// MARK: - Preview

#Preview {
    DriverHUDView(
        cameraManager: CameraManager(),
        faceLandmarkerService: FaceLandmarkerService(),
        eyeAspectCalculator: EyeAspectCalculator(),
        alertManager: AlertManager()
    )
}