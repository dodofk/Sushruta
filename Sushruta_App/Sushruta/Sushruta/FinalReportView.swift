//
//  FinalReport.swift
//  Sushruta
//
//  Created by 莊翔安 on 2022/10/5.
//

import SwiftUI
import GanttisTouch

struct Instrument {
    var name: String
    var MovementImg: String
}

struct FinalReportView: View {
    @State var record: SurgeryRecord
    @State var toolItems = [GanttChartViewItem]()
    @State var dependencies = [GanttChartViewDependency]()
    @State var phaseItems : [GanttChartViewItem] = [
        GanttChartViewItem(row: 0, start: date(0), finish: date(10)),
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
    @State var rowHeaders: [String] = [
        "grasper",
        "hook",
        "scissors",
        "clipper",
        "irrigator",
        "bipolar",
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
                            Text("Benson Starward")
                            Divider()

                            
                            Text(formatter.string(from: record.startTime))
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
                            Text(timeFormatter.string(from: record.startTime))
                            Divider()
                            Text(timeFormatter.string(from: record.endTime ))
                            Divider()
                            Text("\(record.endTime.timeIntervalSince(record.startTime))")
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
                            print("Hello world")
                            for key in record.toolPresentRecord.keys{
                                let idx = toolIndexMap(message: key)
                                let single_tool: [Bool] = record.toolPresentRecord[key] ?? []
                                for index in 0..<(single_tool.count){
                                    if (single_tool[index]){
                                        print(index)
                                        toolItems.append(GanttChartViewItem(row: idx,
                                                                            start: date(index),
                                                                            finish: date(index + 1)))
                                    }
                                }
                            }
                            print(toolItems.count)
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


class ResetModel: ObservableObject {
    func reloadView(){
        objectWillChange.send()
    }
}

//struct FinalReportView_Previews: PreviewProvider {
//    static var previews: some View {
//        FinalReportView()
//    }
//}
