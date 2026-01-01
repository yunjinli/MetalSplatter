#if os(iOS) || os(macOS)

import SwiftUI
import MetalKit

#if os(macOS)
private typealias ViewRepresentable = NSViewRepresentable
#elseif os(iOS)
private typealias ViewRepresentable = UIViewRepresentable
#endif


struct MetalKitSceneView: View {
    var modelIdentifier: ModelIdentifier?
    
    // State for the slider
    @State private var time: Float = 0.0
    @State private var isManualTime: Bool = true

    var body: some View {
        ZStack(alignment: .bottom) {
            MetalView(modelIdentifier: modelIdentifier, manualTime: isManualTime ? time : nil)
                .ignoresSafeArea()

            // UI Overlay
            VStack(spacing: 12) {
                if isManualTime {
                    HStack {
                        Text("Time:")
                            .font(.subheadline)
                            .bold()
                            .foregroundStyle(.white)
                        
                        Text(String(format: "%.2f", time))
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.white)
                            .frame(width: 45, alignment: .leading)

                        Slider(value: $time, in: 0...1)
                            .accentColor(.blue)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                }

                Toggle("Manual Time Control", isOn: $isManualTime)
                    .toggleStyle(.button)
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
            }
            .padding()
            .frame(maxWidth: 400)
        }
    }
}

private struct MetalView: ViewRepresentable {
    var modelIdentifier: ModelIdentifier?
    var manualTime: Float?

    class Coordinator: NSObject {
        var renderer: MetalKitSceneRenderer?
        var startCameraDistance: Float = 0.0
        
#if os(iOS)
        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let renderer = renderer else { return }
            
            let translation = gesture.translation(in: gesture.view)
            
            // Check how many fingers are touching the screen
            if gesture.numberOfTouches == 1 {
                // One finger: Orbit
                let sensitivity: Float = 0.01
                renderer.yaw += Float(translation.x) * sensitivity
                renderer.pitch += Float(translation.y) * sensitivity
                
            } else if gesture.numberOfTouches == 2 {
                // Two fingers: XY pan
                let panSensitivity: Float = 0.005
                renderer.panX += Float(translation.x) * panSensitivity
                renderer.panY -= Float(translation.y) * panSensitivity
            }
            
            // Reset translation so we get incremental updates
            gesture.setTranslation(.zero, in: gesture.view)
        }
        
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let renderer = renderer else { return }

            if gesture.state == .began {
                // Store the current distance when we start pinching
                startCameraDistance = renderer.cameraDistance
            } else if gesture.state == .changed {
                // Calculate new distance.
                let newDistance = startCameraDistance / Float(gesture.scale)
                renderer.cameraDistance = min(max(newDistance, -20.0), -0.5)
            }
        }
#endif // os(iOS)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

#if os(macOS)
    func makeNSView(context: Context) -> MTKView {
        // Use a custom subclass to capture mouse events
        let metalKitView = InteractiveMTKView()
        
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }

        let renderer = MetalKitSceneRenderer(metalKitView)
        context.coordinator.renderer = renderer
        metalKitView.delegate = renderer
        
        // Link the view back to the renderer for input handling
        metalKitView.renderer = renderer

        loadModel(renderer)

        return metalKitView
    }

    func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.renderer?.manualTime = manualTime
        updateView(context.coordinator)
    }
    
    // Custom MTKView subclass to handle Mouse/Trackpad events
    class InteractiveMTKView: MTKView {
        weak var renderer: MetalKitSceneRenderer?
        
        override var acceptsFirstResponder: Bool { true }
        
        // Orbit: Left Mouse Drag
        override func mouseDragged(with event: NSEvent) {
            guard let renderer = renderer else { return }
            let sensitivity: Float = 0.01
            renderer.yaw += Float(event.deltaX) * sensitivity
            renderer.pitch += Float(event.deltaY) * sensitivity
        }
        
        // Pan: Right Mouse Drag (or Control + Click Drag)
        override func rightMouseDragged(with event: NSEvent) {
            guard let renderer = renderer else { return }
            let panSensitivity: Float = 0.005
            renderer.panX += Float(event.deltaX) * panSensitivity
            renderer.panY -= Float(event.deltaY) * panSensitivity
        }
        
        // Pan alternative: Other Mouse Drag (Middle click)
        override func otherMouseDragged(with event: NSEvent) {
            rightMouseDragged(with: event)
        }
        
        // Zoom: Scroll Wheel
        override func scrollWheel(with event: NSEvent) {
            guard let renderer = renderer else { return }
            let scrollSensitivity: Float = 0.5
            // Note: deltaY is usually inverse to distance expectation on scroll
            renderer.cameraDistance += Float(event.deltaY) * scrollSensitivity
            renderer.cameraDistance = min(max(renderer.cameraDistance, -20.0), -0.5)
        }
        
        // Zoom: Pinch Gesture on Trackpad
        override func magnify(with event: NSEvent) {
            guard let renderer = renderer else { return }
            // Magnification is a scale factor (e.g. 1.0 + magnification)
            // We adjust distance inversely to scale
            let scale = Float(1.0 + event.magnification)
            if scale > 0 {
                renderer.cameraDistance /= scale
                renderer.cameraDistance = min(max(renderer.cameraDistance, -20.0), -0.5)
            }
        }
    }
#endif // os(macOS)

#if os(iOS)
    func makeUIView(context: UIViewRepresentableContext<MetalView>) -> MTKView {
        let metalKitView = MTKView()

        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }

        let renderer = MetalKitSceneRenderer(metalKitView)
        context.coordinator.renderer = renderer
        metalKitView.delegate = renderer

        // Add Gesture Recognizers
        let panGesture = UIPanGestureRecognizer(target: context.coordinator,
                                                action: #selector(Coordinator.handlePan(_:)))
        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator,
                                                    action: #selector(Coordinator.handlePinch(_:)))
        panGesture.minimumNumberOfTouches = 1
        panGesture.maximumNumberOfTouches = 2
        metalKitView.addGestureRecognizer(panGesture)
        metalKitView.addGestureRecognizer(pinchGesture)

        loadModel(renderer)
        
        return metalKitView
    }
    
    func updateUIView(_ view: MTKView, context: Context) {
        context.coordinator.renderer?.manualTime = manualTime
        updateView(context.coordinator)
    }
#endif // os(iOS)

    private func loadModel(_ renderer: MetalKitSceneRenderer?) {
        Task {
            do {
                if let modelIdentifier = modelIdentifier {
                    try await renderer?.load(modelIdentifier)
                }
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
    }
    
    private func updateView(_ coordinator: Coordinator) {
        guard let renderer = coordinator.renderer else { return }
        Task {
            do {
                if let modelIdentifier = modelIdentifier {
                    try await renderer.load(modelIdentifier)
                }
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
    }
}

#endif // os(iOS) || os(macOS)
