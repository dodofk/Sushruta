//
//  HistoryMessageView.swift
//  Sushruta
//
//  Created by 莊翔安 on 2022/10/19.
//

import SwiftUI

struct HistoryMessageView: View {
    var body: some View {
        VStack {
            Text("History Message")
                .font(.largeTitle)
            .bold()
            .foregroundColor(.accentColor)
            .padding(20)
            Divider()
                .accentColor(/*@START_MENU_TOKEN@*/Color("AccentColor")/*@END_MENU_TOKEN@*/)
            
            Text("Message")
            Text("Message")
            Text("Message")
            Text("Message")
            Text("Message")
            Text("Message")
            Text("Message")
            Spacer()
        }
    }
}

struct HistoryMessageView_Previews: PreviewProvider {
    static var previews: some View {
        HistoryMessageView()
    }
}
