//
//  FrameView.swift
//  Sushruta
//
//  Created by 劉鈺祥 on 2022/10/5.
//
import SwiftUI
//import UIKit

struct FrameView: View {
    var image: CGImage?
    var bbox: [(CGRect, String)]
    private let label = Text("frame")
    
    var body: some View {
        if let image = image {
            if(bbox.isEmpty){
                Image(decorative: image, scale: 1.0)
                    .resizable()
                    .frame(width: 500, height: 350)
            }else{
                Image(decorative: image, scale: 1.0)
                    .resizable()
                    .frame(width: 500, height: 350)
                    .overlay(
                        ForEach(0..<bbox.count, id: \.self){ i in
                            GeometryReader { geometry in
                                Rectangle()
                                    .path(in: bbox[i].0)
                                    .stroke(Color.red, lineWidth: 2.0)
                                Text(bbox[i].1)
                                    .position(x: bbox[i].0.minX, y: bbox[i].0.minY)
                            }
                        }
                    )
            }
        } else {
            Color.black
        }
    }
}
