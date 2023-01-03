//
//  ContentView.swift
//  Sushruta
//
//  Created by 莊翔安 on 2022/9/27.
//

import SwiftUI
import GanttisTouch


struct ContentViewCamera: View {
    
    @State var isPlaying = false
    @State var showPhaseDetail = false
    @State var showFinalReport = false
    @State var showHistoryMessage = false
    @State var finish = false
    @StateObject private var model = VisionObjectClassificationFrameHandler()
    @State var dependencies = [GanttChartViewDependency]()
    @State var rowHeaders: [String] = [
        "grasper",
        "hook",
        "scissors",
        "clipper",
        "irrigator",
        "monitor",
    ]
    var Phase_Name: [String] = [
        "Preparation",
        "Calot triangle dissection",
        "Clipping and cutting",
        "Galbladder dissection",
        "Galbladder packaging",
        "Cleaning and coagulation",
        "Galbladder retraction",
    ]
    @State var schedule = Schedule.continuous
//    @State var theme = Theme.jewel
    @State var theme = Theme.standard
    
    @State var lastChange = "none"
    @State var dateToAddTo = 2
    @State var finishedTime: Date = Date()
    
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
                    
                    // video information
                    HStack{
                        Text(model.surgeryRecord.surgeryUUid)
                            .font(.system(size: 24))
                        Spacer()
                        if finish {
                            Text("Processing Time: \(finishedTime.timeIntervalSince(model.surgeryRecord.startTime))")
                        }
                        else{
                            Text("Processing Time: \(Date().timeIntervalSince(model.surgeryRecord.startTime))")
                        }
                    }
                    .frame(width: 500)
                    .offset(y: /*@START_MENU_TOKEN@*/14.0/*@END_MENU_TOKEN@*/)
                    
                    
                    FrameView(
                        image: model.frame,
                        bbox: model.bbox
                    ).ignoresSafeArea()
//                    Image("videosample")
//                        .resizable()
//                        .frame(width: 500, height: 350)
//                        .cornerRadius(10.0)
                    
                    
                    
                     // Start and Pause Button
                    Button(action: {
                        isPlaying.toggle()
                        if isPlaying && !finish{
                            model.startRunning()
                        } else {
                            finish = true
                            finishedTime = Date()
                            model.endRunning()
                        }
                    }) {
                        Image(systemName: isPlaying ? "pause" : "play.fill")
                            .padding(3.0)
                            .border(Color.accentColor, width: 2)
                    }
                    .frame(width: 0.0, height: 0.0)

                    .offset(x:220, y:-44)
                    .font(.system(size:34))
                    
                }
                
                
                // phase
                List {
                  Section(header: Text("Phase")) {
                      HStack {
                          Button {
                          } label: {
                              Text(self.Phase_Name[0])
                                  .multilineTextAlignment(.leading)
                                  .padding(7)
                                  .background(Color.blue)
                          }
                          .sheet(isPresented: $showPhaseDetail) {
                              DetailView()
                          }

                      }
                      ForEach(1..<7) { index in
                          HStack {
                              Button {
                                  print("pressing Button\(index+1)")
                                  self.showPhaseDetail.toggle()
                              } label: {
                                  Text(self.Phase_Name[index])
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
                .offset(y:-5)
        

                
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
                                Text("Classification: ")
                                Text(model.label)
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
                            NavigationLink{
                                FinalReportView(record: model.surgeryRecord )
                            } label:{
                                Text("Final Report")
                                    .padding(.horizontal, 15.0)
                                    .padding(.vertical,10)
                                    .foregroundColor(.white)
                                    .background(/*@START_MENU_TOKEN@*//*@PLACEHOLDER=View@*/Color("AccentColor")/*@END_MENU_TOKEN@*/)
                                    .cornerRadius(8)
                                
                                    
                            }
                        } else{
                            Text("Final Report")
                                .padding(.horizontal, 15.0)
                                .padding(.vertical,10)
                                .foregroundColor(.white)
                                .background(/*@START_MENU_TOKEN@*//*@PLACEHOLDER=View@*/Color(hue: 1.0, saturation: 0.0, brightness: 0.834)/*@END_MENU_TOKEN@*/)
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
                        Text("Gantt Chart")
                            .font(.headline)
                        Spacer()
                    }
                    GanttChartView(
                        items: $model.toolItems,
                        dependencies: $dependencies,
                        schedule: schedule,
                        headerRows: [
                            GanttChartHeaderRow(TimeSelector(.days))
                                    ],
                        rowHeight: 50,
                        hourWidth: 21600,

                        scrollableTimeline: TimeRange(from: Time.current.dayStart,
                                                      to: Time.current.adding(hours: 20)),

                        desiredScrollableRowCount: 6,
                        rowHeaders: rowHeaders,
                        rowHeadersWidth: 100,
                        theme: theme,
                        onItemAdded: { item in
                            lastChange = "\(item.label ?? "item") added"
                        },
                        onItemRemoved: { _ in
                            lastChange = "item removed"
                        },
                        onTimeChanged: { item, _ in
                            lastChange = "time updated for \(item.label ?? "item")"
                        },
                        onCompletionChanged: { item, _ in
                            lastChange = "completion updated for \(item.label ?? "item")"
                        },
                        onRowChanged: { item, _ in
                            lastChange = "row updated for \(item.label ?? "item")"
                        }
                    )
                }
                // instrument displacement ratio
                VStack{
                    HStack {
                        Text("Hook movement tracking")
                            .font(.headline)
                        Spacer()
                    }
                    HStack(alignment: .top) {
                        GeometryReader{ geometric in
                            model.surgeryRecord.path().stroke().padding(.trailing)
                        }
                    }
                    .offset(y:-10)
                        
                }
                
            }
        }
    }
    func removeAllDependencies() {
        dependencies.removeAll()
    }
    func changeTheme() {
        theme = theme == .standard ? .jewel : .standard
    }
}

func date(_ second: Int) -> Time {
    return Time().dayStart.adding(seconds: Double(second))
}

struct ContentViewCamera_Previews: PreviewProvider {
    static var previews: some View {
        ContentViewCamera()
    }
}
