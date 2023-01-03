//
//  ContentView.swift
//  Sushruta
//
//  Created by 莊翔安 on 2022/9/27.
//

import SwiftUI
import PhotosUI
import AVKit


struct ContentViewVideo: View {
    
    @State var Start = false
    
    @State var isPlaying = false
    @State var showPhaseDetail = false
    @State var showFinalReport = false
    @State var showHistoryMessage = false
    @State var finish = true
    @State private var selectedItem: [PhotosPickerItem] = []
    @State private var selectedPhotoData: Data?
    @State var player : AVPlayer?
    
    var body: some View {

        VStack(spacing:0){
            Text("Sushruta")
                .font(.largeTitle)
                .foregroundColor(.accentColor)
            
            Divider()
                .padding(.top,10)


            

            HStack{
                // camera
                VStack{
                    if let player{
                        VideoPlayer(player: player).frame(width: 500, height: 300, alignment: .center)
                    } else{
                        HStack{
                            
                            //check button
                            if player != nil{
                                Button {
                                    print("pressing Check")
                                    Start.toggle()
                                } label: {
                                    Text("Start !")
                                        .multilineTextAlignment(.leading)
                                        .padding(.horizontal,7)
                                }
                                .buttonStyle(.borderedProminent)
                            }
                            
                            PhotosPicker(selection: $selectedItem, matching: .any(of: [.images, .videos])) {
                                if selectedItem == nil{
                                    Label("Select a video", systemImage: "video")
                                        .padding(10)
                                        .border(Color("AccentColor"), width: 2)
                                        .cornerRadius(5)
                                }
                                else{
                                    Text("Select another video")
                                }
                            }
                            
                            .onChange(of: selectedItem) { newItem in
                                Task {
                                    guard let item = selectedItem.first else {return}
                                    item.loadTransferable(type: Movie.self){
                                        result in
                                        switch result{
                                        case .success(let movie):
                                            if let movie = movie {
                                                player = AVPlayer(url: movie.url)
                                            } else {
                                                print("movie is nil")
                                            }
                                        case .failure(let failure):
                                            fatalError("\(failure)")
                                        }
                                    }
//                                    if let data = try? await newItem?.loadTransferable(type: Movie.self) {
//                                        selectedPhotoData = data
//                                        print("hello world")
//                                        print(selectedPhotoData)
//                                    }
                                }
                            }
                        }
                    }
                    
                     // Start and Pause Button
//                    Button(action: {
//                        isPlaying.toggle()
//                        if isPlaying {
////                                        player.pause()
//                        } else {
////                                        player.play()
//                        }
//                    }) {
//                        Image(systemName: isPlaying ? "pause" : "play.fill")
//                            .padding(3.0)
//                            .border(Color.accentColor, width: 2)
//                    }
//                    .frame(width: 0.0, height: 0.0)
//                    .font(.system(size:34))
                    
                    
                    
                    //                                VideoPlayer(player: player)
                }
                .frame(minWidth: 0, maxWidth: .infinity, maxHeight: 400)
                
                
                
                
                // phase
                
                VStack {
                    List {
                        Section(header: Text("Phase")) {
                            ForEach(0..<6) { index in
                                HStack {
                                    Button {
                                        print("pressing Button\(index+1)")
                                        self.showPhaseDetail.toggle()
                                    } label: {
                                        Text("Phase-\(index+1)")
                                            .multilineTextAlignment(.leading)
                                            .padding(7)
                                    }
                                    .sheet(isPresented: $showPhaseDetail) {
                                        DetailView()
                                    }
                                    
                                }
                                
                            }
                        }
                    }
                    .listStyle(.plain)
                    .listRowSeparatorTint(.purple)
                .offset(y:-5)
                }
                
                
                
                // instruction and final report
                VStack{
                    // instruction
                    GroupBox(label:
                                HStack{
                        Text("Model Message")
                            .fontWeight(.bold)
                    }){
                        Divider().padding(.vertical, 10)
                        
                        HStack{
                            VStack {
                                Text("This is Message.")
                                Text("This is Message.")
                                Text("This is Message.")
                                Text("This is Message.")
                                Text("This is Message.")
                                Text("This is Message.")
                                Text("This is Message.")
                                Text("This is Message.")
                                Text("This is Message.")
                                Text("This is Message.")
                            }
                            Spacer()
                            
                        }
                    }
                    .frame(width:300, height:300)
                    // two button
                    HStack{
                        
                        Button {
                            print("pressing history instruction")
                            self.showHistoryMessage.toggle()
                        } label: {
                            Text("History Message")
                                .padding(3)
                        }
                        .sheet(isPresented: $showHistoryMessage, content: {
                            HistoryMessageView()
                        })
                        .buttonStyle(.borderedProminent)
                        
                        
                        if finish {
//                            NavigationLink{
//                                FinalReportView()
//                            } label:{
                            Text("Final Report")
                                .padding(.horizontal, 15.0)
                                .padding(.vertical,10)
                                .foregroundColor(.white)
                                .background(Color("AccentColor"))
                                .cornerRadius(8)
                                
                                
//                            }
                        } else{
                            Text("Final Report")
                                .padding(.horizontal, 15.0)
                                .padding(.vertical,10)
                                .foregroundColor(.white)
                                .background(Color(hue: 1.0, saturation: 0.0, brightness: 0.834))
                                .cornerRadius(8)
                            
                            
                        }
                        
                        
                        
                    }
                    .frame(width: 300, height: 50.0)
                    
                }
                
                
            }
            .padding(5.0)
            

            
            
            HStack{
                
                // gannt graph
                VStack{
                    
                    HStack {
                        Text("Gantt Graph")
                            .font(.headline)
                            .padding(0.0)
                            
                        Spacer()
                    }
                    HStack {
                        Image("ganntsample")
                            .resizable()
//                            .frame(width: 550, height: 300)
                            .cornerRadius(10.0)
                        Spacer()
                    }
                    .offset(y:-10)
                        
                }
                Spacer()
                // instrument displacement ratio
                VStack{
                    HStack {
                        Text("Instrument Displacemetn Ratio")
                            .font(.headline)
                        Spacer()
                    }
                    HStack {
                        Image("instrumentdisplacesample")
                            .resizable()
//                            .frame(width: 550, height: 300)
                            .cornerRadius(/*@START_MENU_TOKEN@*/10.0/*@END_MENU_TOKEN@*/)
                        Spacer()
                    }
                    .offset(y:-10)
                        
                }
                
            }
            .padding(5.0)
            .padding(.bottom, 5.0)
        }
    }
}

struct ContentViewVideo_Previews: PreviewProvider {
    static var previews: some View {
        ContentViewVideo()
    }
}





//back-up

//struct DetailView: View{
//
//    @Environment(\.presentationMode) var presentationMode
//
//    var body: some View{
//
//        NavigationView{
//
//            Text("This is the detail for this phase.")
//                .navigationBarItems(trailing: Button(action: {
//                    self.presentationMode.wrappedValue.dismiss()
//                }, label: {
//                    Image(systemName: "chevron.down.circle.fill")
//                        .foregroundColor(.accentColor)
//                }))
//        }
//    }
//}
