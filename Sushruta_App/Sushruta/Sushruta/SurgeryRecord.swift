//
//  SurgeryRecord.swift
//  Sushruta
//
//  Created by 劉鈺祥 on 2022/11/16.
//
import SwiftUI
import Foundation
import GanttisTouch


class SurgeryRecord {
    public var startTime: Date
    public var endTime: Date
    private var hasFinished = false
    public var surgeryUUid: String
    private var Surgeon: String = "Default"
    
    public var hookTrack: [CGRect] = []
    public var grasperTrack: [CGRect] = []
    public var trackList: [CGRect] = []

    
    public var toolPresentRecord: Dictionary<String, [Bool]> = [
        "grasper": [],
        "hook": [],
        "scissors": [],
        "desktop computer": [],
        "mouse, computer mouse,": [],
        "monitor,": [],
    ]
    
    public init(HookTrack: [CGRect], GrasperTrack: [CGRect]){
        self.hookTrack = HookTrack
        self.grasperTrack = GrasperTrack
        self.startTime = Date()
        self.endTime = Date()
        self.surgeryUUid = UUID().uuidString
    }
    
    public init(Surgeon: String = "Default") {
        self.startTime = Date()
        self.endTime = Date()
        self.surgeryUUid = UUID().uuidString
        self.Surgeon = Surgeon
    }
    
    func path(tool: String = "Hook", width: Int = 0, height: Int = 0) -> Path {
        var path = Path()
        switch tool{
        case "Hook":
        trackList = self.hookTrack
        case "Grasper":
        trackList = self.grasperTrack
        default:
        trackList = self.hookTrack
        }
        
        
        if(!trackList.isEmpty){
            path.move(to: CGPoint(x: trackList[0].minX, y: trackList[0].minY))
            for i in 1..<trackList.count {
                path.addLine(to: CGPoint(x: trackList[i].minX, y: trackList[i].minY))
            }
        }
        
        return path
    }
}


