//
//  EyeAspectCalculator.swift
//  Astrateq Vision
//
//  Core fatigue-detection math: Eye Aspect Ratio (EAR) from 6-point eye
//  landmarks, PERCLOS over sliding 30s/60s windows, and driver-state
//  classification. 100% on-device, no persistence of raw landmark data
//  beyond the bounded time window required for PERCLOS.
//
//  Requires: iOS 17+, Swift 6 (strict concurrency)
//

import Foundation
import simd
import CoreGraphics

// MARK: - Landmark Types

/// A single 2D facial landmark point, normalized or pixel-space — the
/// caller is responsible for consistent units since EAR is a ratio and is
/// scale-invariant.
typealias LandmarkPoint = SIMD2<Double>

/// The 6-point eye contour used in the classic Soukupová & Čech EAR
/// formulation: two horizontal corner points and two vertical pairs.
///
/// ```
///        p2      p3
///         •--------•
///   p1 •              • p4
///         •--------•
///        p6      p5
/// ```
struct EyeLandmarks: Sendable, Equatable {
    let outerCorner: LandmarkPoint   // p1
    let upperOuter: LandmarkPoint    // p2
    let upperInner: LandmarkPoint    // p3
    let innerCorner: LandmarkPoint   // p4
    let lowerInner: LandmarkPoint    // p5
    let lowerOuter: LandmarkPoint    // p6

    init(
        outerCorner: LandmarkPoint,
        upperOuter: LandmarkPoint,
        upperInner: LandmarkPoint,
        innerCorner: LandmarkPoint,
        lowerInner: LandmarkPoint,
        lowerOuter: LandmarkPoint
    ) {
        self.outerCorner = outerCorner
        self.upperOuter = upperOuter
        self.upperInner = upperInner
        self.innerCorner = innerCorner
        self.lowerInner = lowerInner
        self.lowerOuter = lowerOuter
    }

    /// Convenience initializer from CGPoint, as commonly returned by
    /// MediaPipe's `FaceLandmarkerResult.faceLandmarks`.
    init(cgPoints points: (CGPoint, CGPoint, CGPoint, CGPoint, CGPoint, CGPoint)) {
        self.outerCorner = LandmarkPoint(points.0.x, points.0.y)
        self.upperOuter  = LandmarkPoint(points.1.x, points.1.y)
        self.upperInner  = LandmarkPoint(points.2.x, points.2.y)
        self.innerCorner = LandmarkPoint(points.3.x, points.3.y)
        self.lowerInner  = LandmarkPoint(points.4.x, points.4.y)
        self.lowerOuter  = LandmarkPoint(points.5.x, points.5.y)
    }
}

/// Canonical MediaPipe Face Mesh (478-point) index sets commonly used for
/// 6-point EAR extraction. Verify against your specific Face Landmarker
/// task/model version before shipping — index sets have shifted across
/// MediaPipe releases.
enum MediaPipeEyeIndices {
    /// Order matches `EyeLandmarks`: outer, upperOuter, upperInner, inner, lowerInner, lowerOuter.
    static let left: [Int]  = [33, 160, 158, 133, 153, 144]
    static let right: [Int] = [362, 385, 387, 263, 373, 380]
}

// MARK: - Driver State

/// Discrete fatigue classification derived from EAR/PERCLOS signals.
/// Ordered by severity for straightforward comparison and alert escalation.
enum DriverState: Int, Sendable, Equatable, Comparable, CaseIterable, CustomStringConvertible {
    case alert = 0
    case drowsy = 1
    case microsleep = 2

    static func < (lhs: DriverState, rhs: DriverState) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    var description: String {
        switch self {
        case .alert: return "Alert"
        case .drowsy: return "Drowsy"
        case .microsleep: return "Microsleep"
        }
    }
}

// MARK: - Fatigue Thresholds

/// Tunable thresholds governing EAR closure detection and PERCLOS-based
/// classification. Defaults are drawn from common driver-drowsiness
/// literature (EAR closure ~0.2, PERCLOS drowsy ~15%, critical ~30%) and
/// should be calibrated against your device's landmark accuracy and
/// target population during QA.
struct FatigueThresholds: Sendable, Equatable {
    /// EAR below this value is treated as "eye closed" for a single frame.
    var earClosureThreshold: Double = 0.21

    /// PERCLOS (0...1) over the short window at/above which the driver is drowsy.
    var perclosDrowsyThreshold: Double = 0.15

    /// PERCLOS (0...1) over the short window at/above which the driver is at microsleep risk.
    var perclosMicrosleepThreshold: Double = 0.30

    /// A single continuous eye-closure event lasting at least this long is
    /// itself classified as a microsleep, independent of PERCLOS.
    var microsleepMinDuration: TimeInterval = 0.5

    /// Long-window PERCLOS threshold used as a secondary confirmation signal
    /// (guards against short bursts of noise triggering sustained "drowsy").
    var longWindowDrowsyThreshold: Double = 0.10

    static let `default` = FatigueThresholds()
}

// MARK: - Fatigue Assessment (output)

/// A single fatigue assessment produced per incoming landmark frame.
struct FatigueAssessment: Sendable, Equatable {
    let timestamp: TimeInterval
    let leftEAR: Double
    let rightEAR: Double
    let averageEAR: Double
    let isEyeClosed: Bool
    let continuousClosureDuration: TimeInterval
    let perclos30: Double
    let perclos60: Double
    let state: DriverState
}

// MARK: - Ring Buffer (O(1) append / evict, no shifting)

/// Fixed-then-growable circular buffer used as the backing store for the
/// EAR sample history. Avoids the O(n) element-shift cost of `Array.removeFirst`
/// on every frame, which matters at 30–60 fps sustained over minutes of driving.
private struct RingBuffer<Element> {
    private var storage: ContiguousArray<Element?>
    private var headIndex = 0
    private(set) var count = 0

    init(capacity: Int) {
        storage = ContiguousArray(repeating: nil, count: Swift.max(capacity, 1))
    }

    var capacity: Int { storage.count }
    var isEmpty: Bool { count == 0 }

    var first: Element? {
        guard count > 0 else { return nil }
        return storage[headIndex]
    }

    mutating func append(_ element: Element) {
        if count == storage.count {
            grow()
        }
        let tailIndex = (headIndex + count) % storage.count
        storage[tailIndex] = element
        count += 1
    }

    @discardableResult
    mutating func removeFirst() -> Element? {
        guard count > 0 else { return nil }
        let element = storage[headIndex]
        storage[headIndex] = nil
        headIndex = (headIndex + 1) % storage.count
        count -= 1
        return element
    }

    /// Iterates elements oldest-to-newest without allocating an intermediate array.
    func forEach(_ body: (Element) -> Void) {
        guard count > 0 else { return }
        var index = headIndex
        for _ in 0..<count {
            if let element = storage[index] {
                body(element)
            }
            index = (index + 1) % storage.count
        }
    }

    private mutating func grow() {
        let newCapacity = Swift.max(storage.count * 2, 1)
        var newStorage = ContiguousArray<Element?>(repeating: nil, count: newCapacity)
        for i in 0..<count {
            newStorage[i] = storage[(headIndex + i) % storage.count]
        }
        storage = newStorage
        headIndex = 0
    }
}

// MARK: - EAR Sample

private struct EARSample: Sendable {
    let timestamp: TimeInterval
    let isClosed: Bool
}

// MARK: - EyeAspectCalculator

/// Actor-isolated fatigue engine. Feed it landmark pairs per frame via
/// `ingest(leftEye:rightEye:timestamp:)`; it returns a complete
/// `FatigueAssessment` synchronously (within actor isolation) with no
/// per-frame heap churn beyond the bounded ring buffer.
///
/// Thread safety: all mutable state (sample history, closure timers) is
/// confined to this actor. `eyeAspectRatio(for:)` is exposed as a
/// `nonisolated static` pure function so callers (e.g. a unit test, or a
/// preview overlay) can compute EAR without actor hops.
actor EyeAspectCalculator {

    // MARK: Configuration

    private let thresholds: FatigueThresholds
    private let shortWindow: TimeInterval
    private let longWindow: TimeInterval

    // MARK: State

    private var history: RingBuffer<EARSample>
    private var closureStartTimestamp: TimeInterval?
    private var lastTimestamp: TimeInterval?

    /// - Parameters:
    ///   - thresholds: Fatigue classification thresholds. Defaults are provided.
    ///   - shortWindowSeconds: PERCLOS short window, typically 30s.
    ///   - longWindowSeconds: PERCLOS long window, typically 60s. Must be >= shortWindowSeconds.
    ///   - expectedFrameRate: Used only to pre-size the ring buffer to avoid
    ///     reallocation during steady-state operation (e.g. 30fps * 60s = 1800).
    init(
        thresholds: FatigueThresholds = .default,
        shortWindowSeconds: TimeInterval = 30,
        longWindowSeconds: TimeInterval = 60,
        expectedFrameRate: Double = 30
    ) {
        precondition(longWindowSeconds >= shortWindowSeconds, "Long window must be >= short window.")
        self.thresholds = thresholds
        self.shortWindow = shortWindowSeconds
        self.longWindow = longWindowSeconds

        // Size with 25% headroom so normal frame-rate jitter never triggers a grow().
        let estimatedCapacity = Int((expectedFrameRate * longWindowSeconds * 1.25).rounded(.up))
        self.history = RingBuffer(capacity: Swift.max(estimatedCapacity, 16))
    }

    // MARK: - Public API

    /// Processes one frame's eye landmarks and returns the current fatigue assessment.
    /// - Parameter timestamp: Monotonic time source (e.g. `CACurrentMediaTime()` or
    ///   the sample buffer's presentation timestamp converted to seconds). Must be
    ///   non-decreasing across calls; out-of-order timestamps are rejected (last
    ///   valid assessment is returned unchanged in that case).
    @discardableResult
    func ingest(
        leftEye: EyeLandmarks,
        rightEye: EyeLandmarks,
        timestamp: TimeInterval
    ) -> FatigueAssessment {
        if let last = lastTimestamp, timestamp < last {
            // Out-of-order frame (e.g. delayed delegate callback). Ignore
            // rather than corrupt the time-ordered ring buffer.
            return currentAssessment(timestamp: last)
        }
        lastTimestamp = timestamp

        let leftEAR = Self.eyeAspectRatio(for: leftEye)
        let rightEAR = Self.eyeAspectRatio(for: rightEye)
        let averageEAR = (leftEAR + rightEAR) * 0.5
        let isClosed = averageEAR < thresholds.earClosureThreshold

        updateClosureTimer(isClosed: isClosed, timestamp: timestamp)

        history.append(EARSample(timestamp: timestamp, isClosed: isClosed))
        pruneHistory(olderThan: timestamp - longWindow)

        let perclos30 = timeWeightedPerclos(windowSeconds: shortWindow, now: timestamp)
        let perclos60 = timeWeightedPerclos(windowSeconds: longWindow, now: timestamp)
        let continuousClosure = closureStartTimestamp.map { timestamp - $0 } ?? 0

        let state = classify(
            perclos30: perclos30,
            perclos60: perclos60,
            continuousClosureDuration: continuousClosure
        )

        return FatigueAssessment(
            timestamp: timestamp,
            leftEAR: leftEAR,
            rightEAR: rightEAR,
            averageEAR: averageEAR,
            isEyeClosed: isClosed,
            continuousClosureDuration: continuousClosure,
            perclos30: perclos30,
            perclos60: perclos60,
            state: state
        )
    }

    /// Resets all accumulated history (e.g. on driver change or trip start).
    func reset() {
        history = RingBuffer(capacity: history.capacity)
        closureStartTimestamp = nil
        lastTimestamp = nil
    }

    // MARK: - Pure EAR math (nonisolated — no actor hop required)

    /// Computes the Eye Aspect Ratio for a single eye:
    /// `EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)`
    ///
    /// Returns 0 if the horizontal eye width is degenerate (coincident
    /// corner points), which prevents NaN/Inf from propagating into PERCLOS.
    nonisolated static func eyeAspectRatio(for eye: EyeLandmarks) -> Double {
        let verticalA = simd.distance(eye.upperOuter, eye.lowerOuter)
        let verticalB = simd.distance(eye.upperInner, eye.lowerInner)
        let horizontal = simd.distance(eye.outerCorner, eye.innerCorner)

        guard horizontal > 1e-9 else { return 0 }
        return (verticalA + verticalB) / (2.0 * horizontal)
    }

    // MARK: - Private helpers

    private func updateClosureTimer(isClosed: Bool, timestamp: TimeInterval) {
        if isClosed {
            if closureStartTimestamp == nil {
                closureStartTimestamp = timestamp
            }
        } else {
            closureStartTimestamp = nil
        }
    }

    private func pruneHistory(olderThan cutoff: TimeInterval) {
        while let oldest = history.first, oldest.timestamp < cutoff {
            history.removeFirst()
        }
    }

    /// Time-weighted PERCLOS: the fraction of wall-clock time within the
    /// window that the eye spent closed, computed from the piecewise-constant
    /// closure state between consecutive samples. This stays accurate even
    /// if the inference frame rate varies (dropped frames, thermal throttling),
    /// unlike a naive closed-sample-count / total-sample-count ratio.
    private func timeWeightedPerclos(windowSeconds: TimeInterval, now: TimeInterval) -> Double {
        guard !history.isEmpty else { return 0 }
        let windowStart = now - windowSeconds

        var closedDuration: TimeInterval = 0
        var coveredDuration: TimeInterval = 0
        var previousTimestamp: TimeInterval?
        var previousClosed = false

        history.forEach { sample in
            if let prevTs = previousTimestamp {
                let segmentStart = Swift.max(prevTs, windowStart)
                let segmentEnd = sample.timestamp
                if segmentEnd > segmentStart {
                    let dt = segmentEnd - segmentStart
                    coveredDuration += dt
                    if previousClosed { closedDuration += dt }
                }
            }
            previousTimestamp = sample.timestamp
            previousClosed = sample.isClosed
        }

        // Extend the final segment up to `now` so the most recent state
        // (typically the current frame) contributes to the ratio.
        if let prevTs = previousTimestamp {
            let segmentStart = Swift.max(prevTs, windowStart)
            if now > segmentStart {
                let dt = now - segmentStart
                coveredDuration += dt
                if previousClosed { closedDuration += dt }
            }
        }

        guard coveredDuration > 0 else { return 0 }
        return min(max(closedDuration / coveredDuration, 0), 1)
    }

    private func classify(
        perclos30: Double,
        perclos60: Double,
        continuousClosureDuration: TimeInterval
    ) -> DriverState {
        // Microsleep: either a single sustained closure event, or short-window
        // PERCLOS crossing the critical threshold.
        if continuousClosureDuration >= thresholds.microsleepMinDuration
            || perclos30 >= thresholds.perclosMicrosleepThreshold {
            return .microsleep
        }

        // Drowsy: short-window PERCLOS elevated, confirmed by a non-trivial
        // long-window trend (guards against a brief noisy burst).
        if perclos30 >= thresholds.perclosDrowsyThreshold
            && perclos60 >= thresholds.longWindowDrowsyThreshold {
            return .drowsy
        }

        return .alert
    }

    /// Reconstructs an assessment for a rejected out-of-order frame without
    /// mutating state, using the most recent EAR values available.
    private func currentAssessment(timestamp: TimeInterval) -> FatigueAssessment {
        let perclos30 = timeWeightedPerclos(windowSeconds: shortWindow, now: timestamp)
        let perclos60 = timeWeightedPerclos(windowSeconds: longWindow, now: timestamp)
        let continuousClosure = closureStartTimestamp.map { timestamp - $0 } ?? 0
        let lastClosed = history.first != nil ? (closureStartTimestamp != nil) : false

        return FatigueAssessment(
            timestamp: timestamp,
            leftEAR: 0,
            rightEAR: 0,
            averageEAR: 0,
            isEyeClosed: lastClosed,
            continuousClosureDuration: continuousClosure,
            perclos30: perclos30,
            perclos60: perclos60,
            state: classify(perclos30: perclos30, perclos60: perclos60, continuousClosureDuration: continuousClosure)
        )
    }
}