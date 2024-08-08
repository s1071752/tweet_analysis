# tweet_analysis
This program is designed to analyze Twitter posts, sentiment, and user engagement.
存放所有資料前處理及資料分析的function

* create_folder: 備份檔案用，檢查資料夾是否存在，如果不存在就創建  
* copy_rawData: 備份檔案用，會複製單一csv檔案到另一個目錄之中  
* create_txt_file: 開小筆記紀錄一些指標用的。若檔案存在就開檔直接繼續新增資料到下一行，不存在則創建檔案  
* read_csv_files_in_directory: 讀取放在目錄之下的所有csv檔案(所以我會把同類型的檔案整理到同一個目錄下做批次處理)  
* get_Col_data: 把需要用到的數個特徵資料列的資料撈出來(因為分析時通常會需要很多特徵)  
* Add_Collumn_to_file: 把計算得出的結果加回創作者的資料檔之中(最後預測會用這個檔案)  
* count_creator_engagement: 計算創作者參與度(公式=參與/粉絲數)  
* count_engagement_Threshold: creator_engagement取平均得到engagement_Threshold  
* judje_creator_success: 根據creator_engagement與engagement_Threshold判斷是否成功    
* str_To_Number: 做follower的前處理  
* Classify_influencerType: 根據網紅定義分類  
  
* detect_urls_hashtags_metions: 檢測此作者的文章是否有hashtag、網址，有的話標註包含次數在檔案中  
* detect_emoji   
* calculate_monthly_post_stability: 計算發文穩定度   
* text_preprocessing: 包含繪文字的話一樣計算出現次數，並將繪文字轉回情緒詞彙  
* get_wordnet_pos: 前處理用。Function to map Penn Treebank POS tag to WordNet POS tag  
* clean_text: 前處理用。移除無法分析的字元並做斷詞  
* Sentiment_Analysis: 計算每一篇貼文的情緒  
* count_rate_of_post_Sentiment: 計算正向情感貼文的比率  
* count_avg_Subjectivity  
  
* train_LDA_model  
* topic_analysis  
* get_topic_and_write  
* symSentSim: 比對大五人格辭典計算OCEAN五大項目得分  
* personality_analysis: 將分數寫入creator_big5.csv檔案中  
* calculate_ocean_avg_scores


