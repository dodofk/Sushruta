//
//  DetailView.swift
//  Sushruta
//
//  Created by 莊翔安 on 2022/10/5.
//

import SwiftUI

struct DetailView: View{

    var body: some View{
        
        VStack{
            
            // phase name
            
            HStack{
                HStack{
                    Spacer()
                    Text("P1 Preparation")
                        .font(.largeTitle)
                        .fontWeight(.black)
                        .padding(10.0)
                        .border(/*@START_MENU_TOKEN@*/Color.black/*@END_MENU_TOKEN@*/, width: 3)
                        .dynamicTypeSize(.xxxLarge)
                        .background(/*@START_MENU_TOKEN@*//*@PLACEHOLDER=View@*/Color(red: 0.712, green: 0.783, blue: 0.756)/*@END_MENU_TOKEN@*/)
                    
                    Spacer()
                        
                    VStack(alignment: .trailing) {
                        Text("Status: Finished")
                            .font(.title)
                            .padding(2.0)
                        Text("Time Consume: 5 mins 11 secs")
                        Text("Start Time: 0 mins 0 secs")
                        Text("End Time: 5 min 11 secs")
                    }
                    .offset(x: /*@START_MENU_TOKEN@*/0.0/*@END_MENU_TOKEN@*/, y: 20)
                    Spacer()
                }
                Spacer()
            }
            .padding(15.0)
            
            Spacer()

            HStack{
                
                Spacer()
                VStack(alignment: .leading) {
                    Text("Final Gannt Graph")
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .offset(x: /*@START_MENU_TOKEN@*/0.0/*@END_MENU_TOKEN@*/, y: /*@START_MENU_TOKEN@*/8.0/*@END_MENU_TOKEN@*/)
                    Image("PhaseDetailGannt")
                        .resizable()
                        .frame(width: 500, height: 250)
                        .cornerRadius(/*@START_MENU_TOKEN@*/10.0/*@END_MENU_TOKEN@*/)
                }
                Spacer()
                
            }
            
            Spacer()
            
            HStack{
                
                Spacer()
                VStack(alignment: .leading) {
                    Text("Tool Usage Distribution")
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .offset(x: /*@START_MENU_TOKEN@*/0.0/*@END_MENU_TOKEN@*/, y: /*@START_MENU_TOKEN@*/8.0/*@END_MENU_TOKEN@*/)
                        
                    Image("PhaseDetailDistribution")
                        .resizable()
                        .frame(width: 500, height: 250)
                        .cornerRadius(/*@START_MENU_TOKEN@*/10.0/*@END_MENU_TOKEN@*/)
                    Spacer()
                }

                Spacer()
                
            }
            
            Spacer()

        }
        .background(Color.accentColor)

    }
}


struct DetailView_Previews: PreviewProvider {
    static var previews: some View {
        DetailView()
    }
}
