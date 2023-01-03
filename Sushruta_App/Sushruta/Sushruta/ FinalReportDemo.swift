//
//  FinalReport.swift
//  Sushruta
//
//  Created by 莊翔安 on 2022/10/5.
//

import SwiftUI
import GanttisTouch


struct FinalReportDemo: View {
    @State var record: SurgeryRecord
    @State var toolItems : [GanttChartViewItem] = [
        GanttChartViewItem(row: 0, start: date(0), finish: date(40)),
        GanttChartViewItem(row: 1, start: date(41), finish: date(45)),
        GanttChartViewItem(row: 0, start: date(43), finish: date(60)),
        GanttChartViewItem(row: 2, start: date(15), finish: date(35)),
        GanttChartViewItem(row: 3, start: date(35), finish: date(36)),
        GanttChartViewItem(row: 3, start: date(43), finish: date(56)),
        GanttChartViewItem(row: 5, start: date(20), finish: date(23)),
        GanttChartViewItem(row: 0, start: date(65), finish: date(72)),
        GanttChartViewItem(row: 1, start: date(62), finish: date(64)),
    ]
    @State var phaseItems : [GanttChartViewItem] = [
        GanttChartViewItem(row: 0, start: date(0), finish: date(4)),
        GanttChartViewItem(row: 1, start: date(5), finish: date(30)),
        GanttChartViewItem(row: 2, start: date(31), finish: date(70)),
        GanttChartViewItem(row: 3, start: date(71), finish: date(210)),
        GanttChartViewItem(row: 4, start:date(211), finish: date(255)),
        GanttChartViewItem(row: 5, start: date(256), finish: date(292)),
        GanttChartViewItem(row: 6, start: date(293), finish: date(325)),
    ]
    
    @State var hookFake: [CGRect] = [
        CGRect(x: 24, y: 76, width: 100, height: 50),
        CGRect(x: 25, y: 85, width: 86, height: 45),
        CGRect(x: 63, y: 100, width: 100, height: 50),
        CGRect(x: 87, y: 87, width: 69, height: 120),
        CGRect(x: 150, y: 120, width: 100, height: 50),
        CGRect(x: 200, y: 151, width: 199, height: 80),
        CGRect(x: 350, y: 180, width: 70, height: 50),
        CGRect(x: 280, y: 70, width: 100, height: 50),
    ]
    
    @State var grasperFake: [CGRect] = [
        CGRect(x: 24, y: 76, width: 100, height: 50),
    ]
    
    @State var dependencies = [GanttChartViewDependency]()
    @State var rowHeaders: [String] = [
        "grasper",
        "hook",
        "scissors",
        "clipper",
        "irrigator",
        "monitor",
    ]
    @State var phaseHeaders: [String] = [
        "Preparation",
        "Calot triangle dissection",
        "Clipping and cutting",
        "Galbladder dissection",
        "Galbladder packaging",
        "Cleaning and coagulation",
        "Galbladder retraction",
    ]
    @State var schedule = Schedule.continuous
    @State var lastChange = "none"
    @State var dateToAddTo = 2
    @ObservedObject var resetmodel = ResetModel()

    private var formatter = DateFormatter()
    private var timeFormatter = DateFormatter()
    
    public init(record: SurgeryRecord){
        self.record = record
        formatter.dateFormat = "yyyy/MM/dd"
        timeFormatter.dateFormat = "HH:mm:ss"
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
    
    @State var imgHeight = 300.0
    @State var imgWidth = 550.0
    let instruments = [
        Instrument(name: "Grasper", MovementImg: "grasper"),
        Instrument(name: "Bipplar", MovementImg: "scissor"),
        Instrument(name: "Hook", MovementImg: "grasper"),
        Instrument(name: "Scissor", MovementImg: "scissor"),
        Instrument(name: "Clipper", MovementImg: "grasper"),
        Instrument(name: "Irrigator", MovementImg: "scissor"),
    ]
    @State private var selectedIndex = 0

    
    var body: some View {
        VStack(spacing: 0) {
            Text("Final Report")
                .font(.largeTitle)
                .bold()
                .foregroundColor(Color.accentColor)
                .padding(.bottom)
            Divider()
            HStack(alignment: .top){
                Spacer()
                Spacer()
                Spacer()
                VStack{
                    
                    //information
                    HStack(spacing: 0){
                        VStack(alignment: .leading){
                            Text("Surgery Name:").bold().foregroundColor(.accentColor)
                            Divider()
                            Text("Surgeon:").bold().foregroundColor(.accentColor)
                            Divider()
                            Text("Date:").bold().foregroundColor(.accentColor)
                            Divider()
                            

                        }  // VStack end
                        VStack(alignment: .trailing){
                            Text("48033C-20220909")
                            Divider()
                            Text("Michael Chen")
                            Divider()

                            
                            Text("2022/12/14")
                            Divider()

                        }  // VStasck end
                        Spacer()
                        
                    }  // Hstack end
                    .font(.title2)
                    HStack(spacing: 0){
                        VStack(alignment: .leading){
                            Text("Start Time:").bold().foregroundColor(.accentColor)
                            Divider()
                            Text("End Times:").bold().foregroundColor(.accentColor)
                            Divider()
                            Text("Time Consumption:").bold().foregroundColor(.accentColor)
                            Divider()
                            

                        }  // VStack end
                        VStack(alignment: .trailing){
                            Text("None")
                            Divider()
                            Text("None")
                            Divider()
                            Text("50:25:16")
                            Divider()

                        }  // VStasck end
                        Spacer()
                        
                    }  // Hstack end
                    .font(.title2)
                    .padding(.bottom)
                    
                    // instrument tracking summary
                    VStack{
                        
                        HStack {
                            Text("Instrument Movement Summary")
                                .font(.system(size: 24))
                                .bold()
                                .padding(.bottom,10)
                                
                            Spacer()
                        }
                        HStack {
                            record.path(tool: instruments[selectedIndex].name).stroke()
                        }
                        .offset(y:-10)
                            
                    }
                    Picker(selection: $selectedIndex) {
                                    ForEach(instruments.indices) { item in
                                        Text(instruments[item].name)
                                    }
                                } label: {
                                    Text("Choose an instrument")
                                }
                                .pickerStyle(.segmented)
                    
                    
                }  // Vstack end
                
                Spacer()
                Spacer()
                
                VStack{
                    
                    // Instrument Usage Summary
                    VStack{
                        
                        HStack {
                            Text("Instrument Usage Summary")
                                .font(.system(size: 24))
                                .bold()
                                .padding(.bottom,10)

                            Spacer()
                        }
                        HStack {
                            GanttChartView(
                                items: $toolItems,
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
                        .onAppear() {
                            resetmodel.reloadView()
                        }
//                        .offset(y:-10)
                        
                    }
                    
                    
                    // Phase Summary
                    VStack{
                        
                        HStack {
                            Text("Phase Summary")
                                .font(.system(size: 24))
                                .bold()
                                .padding(.bottom,10)

                            Spacer()
                        }
                        HStack {
                            GanttChartView(
                                items: $phaseItems,
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
                                rowHeaders: phaseHeaders,
                                rowHeadersWidth: 100,
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
                            ).onAppear() {
                                resetmodel.reloadView()
                            }

                        }
                    }
                    
                    
                    
                }  // Vstack end
            
                Spacer()
                Spacer()
                Spacer()
            }  // HStack end
            .padding(.top)
        Spacer()
        Spacer()
        }  // VStack end
    }
}

//struct FinalReportView_Previews: PreviewProvider {
//    static var previews: some View {
//        FinalReportView()
//    }
//}
