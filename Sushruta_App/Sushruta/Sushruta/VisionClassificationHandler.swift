//
//  VisionClassificationFrameHandler.swift
//  Sushruta
//
//  Created by 劉鈺祥 on 2022/10/5.
//

import Vision
import AVFoundation
import UIKit
import GanttisTouch

class VisionObjectClassificationFrameHandler : FrameHandler {
    
    private var flag: Bool = false
    private var time: Time = Time()
    private var requests = [VNRequest]()
    @Published public var surgeryRecord: SurgeryRecord = SurgeryRecord()
    private var date2Add = 0
    
    override init(){
        super.init()
        setupVision()
//        session.startRunning()
    }
    
    func startRunning(){
        session.startRunning()
        surgeryRecord = SurgeryRecord()
    }
    
    func endRunning(){
        session.stopRunning()
        surgeryRecord.endTime = Date()
    }

    
    @discardableResult
    func setupVision() -> NSError? {
        let error: NSError! = nil
        
        guard let modelURL = Bundle.main.url(forResource: "Resnet50FP16", withExtension: "mlmodelc") else{
            return NSError(domain: "VisionObjectDetectionViewController", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
        }
        guard let detectionModelURL = Bundle.main.url(forResource: "yolov5s", withExtension: "mlmodelc") else{
            return NSError(domain: "VisionObjectDetectionViewController", code: -1, userInfo: [NSLocalizedDescriptionKey: "Detection file is missing"])
        }
        do {
            let classificationModel = try VNCoreMLModel(for: MLModel(contentsOf:  modelURL))
            let detectionModel = try VNCoreMLModel(for: MLModel(contentsOf: detectionModelURL))
            let objectClassification = VNCoreMLRequest(model: classificationModel, completionHandler: { (request, error) in
                DispatchQueue.main.async(execute: {
                    // perform all the UI updates on the main queue
                    if let results = request.results {
                        self.drawVisionRequestResults(results)
                    }
                })
            })
            let objectDetection = VNCoreMLRequest(model: detectionModel, completionHandler: { (request, error) in
                DispatchQueue.main.async(execute: {
                    // perform all the UI updates on the main queue
                    if let results = request.results {
                        self.drawVisionDetectionResults(results)
                    }
                })
            })
            objectClassification.imageCropAndScaleOption = .scaleFill
            objectDetection.imageCropAndScaleOption = .scaleFill
            self.requests = [objectClassification, objectDetection]

        } catch let error as NSError{
            print("Model loading went wrong: \(error)")
        }
        
        return error
    }
    
    func drawVisionRequestResults(_ results: [Any]){
        var result: [String] = []
        let threshold: Float = 0.25
        var toolPresent = [
            "grasper": false,
            "hook": false,
            "scissors": false,
            "desktop computer": false,
            "mouse, computer mouse,": false,
            "monitor,": false,
        ]
        
        for observation in results where observation is VNClassificationObservation{
            guard let clsObservation = observation as? VNClassificationObservation else {
                continue
            }
            if(clsObservation.confidence > threshold){
                result.append("\(clsObservation.identifier),")
                if(toolPresent["\(clsObservation.identifier),"] != nil){
                    toolPresent["\(clsObservation.identifier),"] = true
                }
            }
        }
        for key in toolPresent.keys{
            surgeryRecord.toolPresentRecord[key]?.append(toolPresent[key] ?? false)
        }
        updatePredictionLabel(result)
    }
    
    func drawVisionDetectionResults(_ results: [Any]){
        let threshold: Float = 0.3
        var candidate: [(CGRect, String)] = []
        
        let tools = surgeryRecord.toolPresentRecord.keys
        for observation in results where observation is VNRecognizedObjectObservation {
            guard let detectObservation = observation as? VNRecognizedObjectObservation else {
                continue
            }
            
            let topLabelObservation = detectObservation.labels[0]
            let objectBounds = getConvertedRect(boundingBox: detectObservation.boundingBox, width: 500, height: 300)
//            let objectBounds = VNImageRectForNormalizedRect(detectObservation.boundingBox, Int(500), Int(350))
            if(topLabelObservation.confidence >= threshold){
                candidate.append((objectBounds, topLabelObservation.identifier))
                if(topLabelObservation.identifier == "keyboard"){
                    surgeryRecord.hookTrack.append(objectBounds)
                } else if (topLabelObservation.identifier == "laptop"){
                    surgeryRecord.grasperTrack.append(objectBounds)
                }
            }
        }
        
        updateBoundingBox(candidate)
    }

    func toolIndexMap(message: String) -> Int{
        switch message{
        case "grasper":
            return 0
        case "hook":
            return 1
        case "scissors":
            return 2
        case "desktop computer,":
            return 3
        case "mouse, computer mouse,":
            return 4
        case "monitor,":
            return 5
        default:
            return -1
        }
    }
    
    func updateTool(_ message: [String]){
        for label in message{
            let index = toolIndexMap(message: label)
            if index == -1{
                continue
            }
            self.toolItems.append(
                GanttChartViewItem(row: index,
                                   start: date(self.date2Add),
                                   finish: date(self.date2Add + 1)))
        }
        self.date2Add += 1
    }
    
    func updatePredictionLabel(_ message: [String]) {
        DispatchQueue.main.async {
            [unowned self] in
            updateTool(message)
            self.label = message.joined(separator: "\n")
        }
    }
    
    func updateBoundingBox(_ rect: [(CGRect, String)]){
        DispatchQueue.main.async{
            [unowned self] in
            self.bbox = rect
        }
    }
    
    func updateDetectionLabel(_ message: [String]) {
        DispatchQueue.main.async {
            [unowned self] in
            self.detectionlabel = message.joined(separator: "\n")
        }
    }
    
    func getConvertedRect(boundingBox: CGRect, width: Int, height: Int) -> CGRect{
        let newBbox = CGRect(
            x: boundingBox.origin.x,
            y: 1 - boundingBox.origin.y - boundingBox.height,
            width: boundingBox.width,
            height: boundingBox.height
        )
        let convertedRect = VNImageRectForNormalizedRect(newBbox, width, height)
        
        return convertedRect
    }
}


extension VisionObjectClassificationFrameHandler {
    override func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection){
        
        guard let cgImage = imageFromSampleBuffer(sampleBuffer: sampleBuffer) else {return}
        
        DispatchQueue.main.async {
            [unowned self] in
            self.frame = cgImage
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return}
    
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
}

