# Sushruta

台大資管專題 - Sushruta

Sushruta 是一個手術輔助系統，目標是為了解決膽囊切除術時
因為醫師的技術良莠不齊導致手術品質不良，甚至有併發症以及後遺症的
可能。

我們的系統有三個主要的功能：
* 追蹤器具軌跡
* 辨識三元組(器具、目標、動作)
* 階段辨識

其中我們系統能產生出器具移動軌跡、器具甘特圖等關鍵資訊輔助手術系統，其中甘特圖等資訊可以反應出手術的品質和需要注意的事項，也能幫助醫生去檢討哪裡可以改善。

## Sushruta Model

Model 程式碼包括所有有關模型的程式碼，包含針對三元組辨識模型的參數調整等實驗，和所提出基於 Attention 分辨目標的模型。 其中使用 Python、Pytorch 作為主要語言來撰寫。


## Sushruta App

Sushruta App 包含使用了 Swift 撰寫的 IOS IPad App，讓醫生可以方便的操作和使用並且紀錄，App 可以實時產生甘特圖、器具軌跡圖等讓醫生進行參考。

## Report

Report.pdf 是資管專題報告的投影片，可以參考投影片中的內容知道我們這專案的主要價值和貢獻。

## Contact
### Project member
- 歐崇愷: b08303028@ntu.edu.tw 
- 劉鈺祥: b08705024@ntu.edu.tw 
- 莊翔安: b08303028@ntu.edu.tw 
- 陳旻浚: b08705051@ntu.edu.tw 

### Advisor
李家岩教授: chiayenlee@ntu.edu.tw

### Mentor
歐子毓: a1225johnny@gmail.com

Welcome to contact us contact us with above information!