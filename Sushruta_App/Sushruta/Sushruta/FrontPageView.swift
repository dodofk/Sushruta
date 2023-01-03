//
//  FrontPageView.swift
//  Sushruta
//
//  Created by 莊翔安 on 2022/10/10.
//

import SwiftUI



struct FrontPageView: View {
    @Environment(\.presentationMode) var presentationMode
    @State var surgeryRecord =  SurgeryRecord(
    HookTrack: [
        CGRect(x: 124, y: 55, width: 100, height: 50),
        CGRect(x: 132, y: 72, width: 86, height: 45),
        CGRect(x: 126, y: 66, width: 70, height: 50),
        CGRect(x: 134, y: 72, width: 86, height: 45),
        CGRect(x: 263, y: 200, width: 100, height: 50),
        CGRect(x: 287, y: 197, width: 69, height: 120),
        CGRect(x: 250, y: 150, width: 100, height: 50),
        CGRect(x: 180, y: 135, width: 199, height: 80),
        CGRect(x: 166, y: 75, width: 70, height: 50),
        CGRect(x: 146, y: 82, width: 100, height: 50),
        CGRect(x: 120, y: 230, width: 100, height: 45),
        CGRect(x: 125, y: 245, width: 100, height: 45),
        CGRect(x: 118, y: 240, width: 100, height: 45),
        CGRect(x: 80, y: 280, width: 100, height: 45),
        CGRect(x: 92, y: 263, width: 100, height: 45),
    ], GrasperTrack: [
        CGRect(x: 24, y: 76, width: 100, height: 50),
        CGRect(x: 25, y: 85, width: 86, height: 45),
        CGRect(x: 63, y: 100, width: 100, height: 50),
        CGRect(x: 65, y: 96, width: 70, height: 50),
        CGRect(x: 63, y: 98, width: 70, height: 50),
        CGRect(x: 87, y: 87, width: 69, height: 120),
        CGRect(x: 150, y: 120, width: 100, height: 50),
        CGRect(x: 200, y: 151, width: 199, height: 80),
        CGRect(x: 230, y: 165, width: 70, height: 50),
        CGRect(x: 350, y: 180, width: 70, height: 50),
        CGRect(x: 280, y: 120, width: 100, height: 50),
        CGRect(x: 283, y: 115, width: 100, height: 55),
        CGRect(x: 250, y: 200, width: 100, height: 45),
        CGRect(x: 423, y: 205, width: 100, height: 45),
        CGRect(x: 399, y: 207, width: 100, height: 45),
    ])
    @State var showHistoryData = false
    @State var shouldPresentActionScheet = false
    @State var goContentViewCamera = false
    @State var goContentViewVideo = false
        
    var body: some View {
        NavigationView {
            
            //switching page
            if goContentViewVideo{
                NavigationLink("", destination: ContentViewVideo(), isActive: $goContentViewVideo)
            }
            if goContentViewCamera{
                NavigationLink("", destination: ContentViewCamera(), isActive: $goContentViewCamera)
            }
            
            
            VStack(spacing: 0) {
                HStack {
                    
                    Button(action:{
                        print("pressing logo")
                    }) {
                        HStack(alignment: .bottom){
                            
                            Image("logo")
                                .padding(.leading, 5.0)
                            Text("Sushruta")
                                .font(.largeTitle)
        //                        .foregroundColor(Color.white)
                        }
                    }
                    .padding(.horizontal,5)
                    Spacer()

                }
                .padding(.vertical)
                .frame(maxWidth: .infinity)
                .accentColor(Color.black)
                .background(Color.accentColor)
                
                
                List {
                    ForEach(1..<2) { index in
                        NavigationLink{
//                            FinalReportView()
                        } label: {
                            HStack {
                                Image("history-data-photo-\(index)")
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 100, height: 50)
                                    .cornerRadius(10)

                                Text("48033C-2022090\(index)")
                                    .bold()
                            }
                        }
                    }
                    NavigationLink{
                        FinalReportDemo(record: surgeryRecord)
                    } label: {
                        HStack {
                            Image("history-data-photo-2")
                                .resizable()
                                .scaledToFill()
                                .frame(width: 100, height: 50)
                                .cornerRadius(10)

                            Text("48033C-202209002")
                                .bold()
                        }
                    }
                }
//                List {
//                  Section(header: Text("History Data")) {
//                  ForEach(0..<6) { index in
//                      HStack {
//                          Button {
//                              print("pressing history data\(index+1)")
//                              self.showHistoryData.toggle()
//                          } label: {
//                              Text("2022/01/0\(index+1)")
//                                  .multilineTextAlignment(.leading)
//                                  .padding(8)
//                          }
//                          .sheet(isPresented: $showHistoryData) {
//                              FinalReportView()
//                          }
//
//                      }
//
//                    }
//                  }
//                }  // List end
                
                HStack {
                    Spacer()
                    
//                    NavigationLink{
//                        ContentView()
//                    } label: {
//                        Text("New Video")
//                            .font(.system(size: 40))
//                            .fontWeight(.bold)
//                            .font(.title)
//                            .foregroundColor(.accentColor)
//                            .padding()
//                            .overlay(
//                                RoundedRectangle(cornerRadius: 20)
//                                    .stroke(Color.accentColor, lineWidth: 8)
//                            )
//                            .background(Color.white)
//                            .cornerRadius(20)
//                    }
                    
                    Text("New Video")
                        .font(.system(size: 40))
                        .fontWeight(.bold)
                        .font(.title)
                        .foregroundColor(.accentColor)
                        .padding()
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Color.accentColor, lineWidth: 8)
                        )
                        .background(Color.white)
                        .cornerRadius(20)
                        .onTapGesture {shouldPresentActionScheet = true}
                        .actionSheet(isPresented: $shouldPresentActionScheet) { () -> ActionSheet in
                            ActionSheet(title: Text("Choose mode"), message: Text("Please choose your preferred mode to start."), buttons: [ActionSheet.Button.default(Text("Camera"), action: {
                                goContentViewCamera = true
                            }), ActionSheet.Button.default(Text("Video Library"), action: {
                                goContentViewVideo = true
                            }), ActionSheet.Button.cancel()])
                        }
                    
                    
                    
                    Spacer()
                }
                .background(/*@START_MENU_TOKEN@*//*@PLACEHOLDER=View@*/Color(red: 0.95, green: 0.945, blue: 0.971)/*@END_MENU_TOKEN@*/)  // HStack end
                
                Spacer()
                Spacer()
                

                
                
            }  // VStack end
            .ignoresSafeArea(.all, edges: .bottom)
        }
        .navigationViewStyle(StackNavigationViewStyle())
        
    }  // body View end
}  // View end

struct FrontPageView_Previews: PreviewProvider {
    static var previews: some View {
        FrontPageView()
    }
}
