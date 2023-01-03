//
//  Surgery.swift
//  Sushruta
//
//  Created by 莊翔安 on 2022/10/11.
//

import Foundation
import SwiftUI

struct Landmark: Hashable, Codable {
    var name: String

    private var imageName: String
    var image: Image {
        Image(imageName)
    }

}


