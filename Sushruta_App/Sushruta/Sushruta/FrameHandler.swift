//
//  FrameHandler.swift
//  Sushruta
//
//  Created by 劉鈺祥 on 2022/10/5.
//

import AVFoundation
import CoreImage
import GanttisTouch

class FrameHandler : NSObject, ObservableObject {
    @Published var frame: CGImage?
    @Published var label = ""
    @Published var bbox: [(CGRect, String)] = []
    @Published var detectionlabel = ""
    @Published var toolItems = [GanttChartViewItem]()

    
    
    var bufferSize: CGSize = .zero
    
    private var permissionGranted = false
    internal let session = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    private let context = CIContext()
    
    override init() {
        super.init()
        checkPermission()
        self.setupAVCapture()
    }
    
    func checkPermission(){
        switch AVCaptureDevice.authorizationStatus(for: .video){
        case .authorized:
            permissionGranted = true
        case .notDetermined:
            requestPermissions()
        default:
            permissionGranted = false
        }
    }
    
    func requestPermissions(){
        AVCaptureDevice.requestAccess(for: .video) {
            [unowned self] granted in
            self.permissionGranted = granted
        }
    }
    
    func startCaptureSession(){
        self.session.startRunning()
    }
    
    func setupAVCapture() {
        var deviceInput: AVCaptureInput!
        
        
        let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
        }catch{
            print("Could not create video device input: \(error)")
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        
        guard session.canAddInput(deviceInput) else{
            print("Cound not add video device input to the seesion")
            session.commitConfiguration()
            return
        }
        session.addInput(deviceInput)
        
        if session.canAddOutput(videoDataOutput){
            print("Testing")
            session.addOutput(videoDataOutput)
            
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else {
            print("Could not add video data output to the session")
            session.commitConfiguration()
            return
        }
        let captureConnection = videoDataOutput.connection(with: .video)
        
        captureConnection?.isEnabled = true
        do{
            try videoDevice!.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice?.activeFormat.formatDescription)!)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice!.unlockForConfiguration()
        } catch {
            print(error)
        }
        session.commitConfiguration()
    }
}

extension FrameHandler: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection){
        guard let cgImage = imageFromSampleBuffer(sampleBuffer: sampleBuffer) else {return}
        
        DispatchQueue.main.async {
            [unowned self] in
            self.frame = cgImage
        }
    }
    
    internal func imageFromSampleBuffer(sampleBuffer: CMSampleBuffer) -> CGImage? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return nil}
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {return nil}
        
        return cgImage
    }
}
