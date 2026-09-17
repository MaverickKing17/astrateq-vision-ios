//
//  FaceLandmarkerService.swift
//  Astrateq Vision
//
//  Actor-isolated wrapper around Google MediaPipe Tasks Vision's
//  FaceLandmarker, configured for low-latency .liveStream inference.
//  Converts CVPixelBuffer frames from CameraManager into 478-point 3D
//  facial landmarks, delivered asynchronously so inference never blocks
//  the capture pipeline or the main/UI thread.
//
//  Dependency: MediaPipeTasksVision (CocoaPods: "MediaPipeTasksVision",
//  or Swift Package: https://github.com/google/mediapipe)
//  Model asset: bundle a `face_landmarker.task` file (from the MediaPipe
//  model zoo) into the app target's "Copy Bundle Resources" build phase.
//
//  Requires: iOS 17+, Swift 6 (strict concurrency)
//

import Foundation
import CoreVideo
import CoreGraphics
import UIKit
import MediaPipeTasksVision
import os.log

// MARK: - Errors

enum FaceLandmarkerServiceError: Error, LocalizedError, Sendable {
    case modelFileNotFound(String)
    case initializationFailed(String)
    case notConfigured
    case imageConversionFailed(String)
    case detectionRequestFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelFileNotFound(let detail):
            return "FaceLandmarker model asset not found: \(detail)"
        case .initializationFailed(let detail):
            return "FaceLandmarker failed to initialize: \(detail)"
        case .notConfigured:
            return "FaceLandmarkerService.start() must succeed before calling detectAsync(_:)."
        case .imageConversionFailed(let detail):
            return "Failed to convert CVPixelBuffer to MPImage: \(detail)"
        case .detectionRequestFailed(let detail):
            return "MediaPipe detection request failed: \(detail)"
        }
    }
}

// MARK: - Configuration

/// Tunable MediaPipe FaceLandmarker options. Defaults favor single-driver
/// tracking with balanced precision/latency for real-time fatigue monitoring.
struct FaceLandmarkerConfiguration: Sendable {
    var modelResourceName: String = "face_landmarker"
    var modelResourceExtension: String = "task"
    var numFaces: Int = 1
    var minFaceDetectionConfidence: Float = 0.5
    var minFacePresenceConfidence: Float = 0.5
    var minTrackingConfidence: Float = 0.5
    var outputFaceBlendshapes: Bool = false
    var outputFacialTransformationMatrixes: Bool = false

    static let `default` = FaceLandmarkerConfiguration()
}

// MARK: - Output

/// One frame's worth of detected face landmarks, converted to pixel-space
/// coordinates (not MediaPipe's raw normalized [0,1] output) so downstream
/// consumers like `EyeAspectCalculator` get geometrically correct ratios
/// regardless of the source buffer's aspect ratio.
struct FaceLandmarkerOutput: Sendable {
    let timestampMs: Int
    /// Pixel-space landmarks: x, y in pixels of the source buffer; z follows
    /// MediaPipe's convention of roughly the same scale as x, relative depth
    /// (more negative = closer to camera).
    let landmarks: [SIMD3<Double>]
    let imageWidth: Int
    let imageHeight: Int

    /// Convenience extraction of the 6-point eye contour used by
    /// `EyeAspectCalculator`. Returns `nil` if the landmark set doesn't
    /// contain the expected indices (e.g. mismatched model version).
    var leftEye: EyeLandmarks? { Self.eye(from: landmarks, indices: MediaPipeEyeIndices.left) }
    var rightEye: EyeLandmarks? { Self.eye(from: landmarks, indices: MediaPipeEyeIndices.right) }

    private static func eye(from landmarks: [SIMD3<Double>], indices: [Int]) -> EyeLandmarks? {
        guard indices.allSatisfy({ $0 >= 0 && $0 < landmarks.count }) else { return nil }
        let points = indices.map { LandmarkPoint(landmarks[$0].x, landmarks[$0].y) }
        return EyeLandmarks(
            outerCorner: points[0],
            upperOuter: points[1],
            upperInner: points[2],
            innerCorner: points[3],
            lowerInner: points[4],
            lowerOuter: points[5]
        )
    }
}

// MARK: - FaceLandmarkerService

/// Actor-isolated MediaPipe FaceLandmarker wrapper. Owns the `FaceLandmarker`
/// instance and its live-stream delegate; exposes results as an
/// `AsyncStream` so consumers pull them with normal Swift Concurrency rather
/// than implementing a delegate themselves.
///
/// Threading: `detectAsync(pixelBuffer:timestampMs:)` submits work and
/// returns immediately — it does not block on inference. MediaPipe invokes
/// its live-stream delegate on an internal worker thread; the bridge below
/// forwards that callback into this actor, so all state mutation here is
/// serialized and thread-safe. Callers should invoke `detectAsync` from a
/// background `Task`, never from the main actor, to keep the call site
/// itself non-blocking even though the call is technically async-return-fast.
actor FaceLandmarkerService {

    // MARK: State

    private let configuration: FaceLandmarkerConfiguration
    private var faceLandmarker: FaceLandmarker?
    private var delegateBridge: LiveStreamResultBridge?

    /// MediaPipe's live-stream mode requires strictly increasing timestamps.
    private var lastTimestampMs: Int = -1

    /// Dimensions of the buffer most recently submitted, used to convert the
    /// next delegate callback's normalized landmarks to pixel space. Safe
    /// under actor isolation: both the write (in `detectAsync`) and the read
    /// (in `handleResult`) are serialized on this actor, and camera
    /// resolution is constant for the life of a session.
    private var lastImageWidth: Int = 0
    private var lastImageHeight: Int = 0

    private(set) var resultStream: AsyncStream<FaceLandmarkerOutput>!
    private var resultContinuation: AsyncStream<FaceLandmarkerOutput>.Continuation!

    private(set) var errorStream: AsyncStream<FaceLandmarkerServiceError>!
    private var errorContinuation: AsyncStream<FaceLandmarkerServiceError>.Continuation!

    private let logger = Logger(subsystem: "com.astrateq.vision", category: "FaceLandmarkerService")

    init(configuration: FaceLandmarkerConfiguration = .default) {
        self.configuration = configuration

        var resultCont: AsyncStream<FaceLandmarkerOutput>.Continuation!
        let rStream = AsyncStream<FaceLandmarkerOutput> { resultCont = $0 }
        self.resultStream = rStream
        self.resultContinuation = resultCont

        var errCont: AsyncStream<FaceLandmarkerServiceError>.Continuation!
        let eStream = AsyncStream<FaceLandmarkerServiceError> { errCont = $0 }
        self.errorStream = eStream
        self.errorContinuation = errCont
    }

    deinit {
        resultContinuation?.finish()
        errorContinuation?.finish()
    }

    // MARK: - Lifecycle

    /// Locates the bundled `.task` model asset and configures MediaPipe for
    /// live-stream inference. Idempotent — safe to call multiple times.
    func start() throws {
        guard faceLandmarker == nil else { return }

        guard let modelPath = Bundle.main.path(
            forResource: configuration.modelResourceName,
            ofType: configuration.modelResourceExtension
        ) else {
            let error = FaceLandmarkerServiceError.modelFileNotFound(
                "\(configuration.modelResourceName).\(configuration.modelResourceExtension) " +
                "not found in main bundle. Confirm it's included in Copy Bundle Resources."
            )
            logger.error("\(error.localizedDescription, privacy: .public)")
            throw error
        }

        let options = FaceLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
        options.numFaces = configuration.numFaces
        options.minFaceDetectionConfidence = configuration.minFaceDetectionConfidence
        options.minFacePresenceConfidence = configuration.minFacePresenceConfidence
        options.minTrackingConfidence = configuration.minTrackingConfidence
        options.outputFaceBlendshapes = configuration.outputFaceBlendshapes
        options.outputFacialTransformationMatrixes = configuration.outputFacialTransformationMatrixes

        let bridge = LiveStreamResultBridge(service: self)
        options.faceLandmarkerLiveStreamDelegate = bridge
        self.delegateBridge = bridge

        do {
            self.faceLandmarker = try FaceLandmarker(options: options)
        } catch {
            let wrapped = FaceLandmarkerServiceError.initializationFailed(error.localizedDescription)
            logger.error("FaceLandmarker init failed: \(error.localizedDescription, privacy: .public)")
            throw wrapped
        }

        logger.info("FaceLandmarkerService configured: liveStream mode, numFaces=\(self.configuration.numFaces).")
    }

    /// Tears down the MediaPipe instance and finishes both output streams.
    /// Call when monitoring stops (e.g. trip ended, app backgrounded).
    func shutdown() {
        faceLandmarker = nil
        delegateBridge = nil
        lastTimestampMs = -1
        lastImageWidth = 0
        lastImageHeight = 0
        resultContinuation.finish()
        errorContinuation.finish()
        logger.info("FaceLandmarkerService shut down.")
    }

    // MARK: - Detection

    /// Submits one frame for asynchronous detection. Returns as soon as the
    /// frame is handed to MediaPipe's internal queue — it does not wait for
    /// inference to complete. Results (or errors) arrive later via
    /// `resultStream` / `errorStream`.
    ///
    /// - Parameters:
    ///   - pixelBuffer: A BGRA `CVPixelBuffer`, typically from
    ///     `CameraManager.frameStream`.
    ///   - timestampMs: Monotonic timestamp in milliseconds. Values that are
    ///     not strictly greater than the previous call are coalesced to
    ///     `lastTimestampMs + 1` rather than thrown, since MediaPipe's
    ///     live-stream mode requires strict ordering and silently dropping
    ///     a frame here would create a visible gap in the fatigue signal.
    ///   - orientation: Orientation of the buffer's contents. Defaults to
    ///     `.up` — correct when `CameraManager`'s capture connection has
    ///     already applied rotation/mirroring, as in this codebase. Adjust
    ///     if you change that configuration.
    func detectAsync(
        pixelBuffer: CVPixelBuffer,
        timestampMs: Int,
        orientation: UIImage.Orientation = .up
    ) throws {
        guard let faceLandmarker else {
            throw FaceLandmarkerServiceError.notConfigured
        }

        let effectiveTimestamp = timestampMs > lastTimestampMs ? timestampMs : lastTimestampMs + 1
        lastTimestampMs = effectiveTimestamp
        lastImageWidth = CVPixelBufferGetWidth(pixelBuffer)
        lastImageHeight = CVPixelBufferGetHeight(pixelBuffer)

        let image: MPImage
        do {
            image = try MPImage(pixelBuffer: pixelBuffer, orientation: orientation)
        } catch {
            let wrapped = FaceLandmarkerServiceError.imageConversionFailed(error.localizedDescription)
            logger.error("\(wrapped.localizedDescription, privacy: .public)")
            throw wrapped
        }

        do {
            try faceLandmarker.detectAsync(image: image, timestampInMilliseconds: effectiveTimestamp)
        } catch {
            let wrapped = FaceLandmarkerServiceError.detectionRequestFailed(error.localizedDescription)
            logger.error("\(wrapped.localizedDescription, privacy: .public)")
            throw wrapped
        }
    }

    // MARK: - Delegate callback entry point

    /// Invoked by `LiveStreamResultBridge` on MediaPipe's result callback.
    /// Runs on this actor, so mutation of `lastImageWidth`/`lastImageHeight`
    /// reads and stream yields are all serialized safely.
    fileprivate func handleResult(
        _ result: FaceLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        if let error {
            let wrapped = FaceLandmarkerServiceError.detectionRequestFailed(error.localizedDescription)
            logger.error("Detection callback error @t=\(timestampInMilliseconds): \(error.localizedDescription, privacy: .public)")
            errorContinuation.yield(wrapped)
            return
        }

        guard let result, let face = result.faceLandmarks.first, !face.isEmpty else {
            // No face detected this frame (e.g. driver looked away). Not an
            // error condition — simply nothing to yield. Downstream fatigue
            // logic should treat a gap in the result stream as "face not
            // visible" rather than "eyes closed".
            return
        }

        let width = Double(lastImageWidth)
        let height = Double(lastImageHeight)
        let points: [SIMD3<Double>] = face.map { landmark in
            SIMD3<Double>(
                Double(landmark.x) * width,
                Double(landmark.y) * height,
                Double(landmark.z) * width
            )
        }

        let output = FaceLandmarkerOutput(
            timestampMs: timestampInMilliseconds,
            landmarks: points,
            imageWidth: lastImageWidth,
            imageHeight: lastImageHeight
        )
        resultContinuation.yield(output)
    }
}

// MARK: - Live-Stream Delegate Bridge

/// MediaPipe's `FaceLandmarkerLiveStreamDelegate` must be a reference type
/// conforming to `AnyObject`, so it cannot be the actor itself. This bridge
/// receives the callback on MediaPipe's internal thread and forwards it into
/// `FaceLandmarkerService` via a `Task`, keeping all result-handling state
/// changes confined to the actor.
private final class LiveStreamResultBridge: NSObject, FaceLandmarkerLiveStreamDelegate, @unchecked Sendable {
    private weak var service: FaceLandmarkerService?

    init(service: FaceLandmarkerService) {
        self.service = service
    }

    func faceLandmarker(
        _ faceLandmarker: FaceLandmarker,
        didFinishDetection result: FaceLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        guard let service else { return }
        Task {
            await service.handleResult(result, timestampInMilliseconds: timestampInMilliseconds, error: error)
        }
    }
}