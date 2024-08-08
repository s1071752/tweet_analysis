# 讀取檔案
import os 
import pandas as pd
import numpy as np
# 使用Nature Language Tool Kit (NLTK)處理文本
import nltk  
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
'''
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
'''
# 文本處理
import string
from textblob import TextBlob
import emoji
import re
#LDA
import gensim
from gensim import corpora, models
from pprint import pprint
from gensim.models import LdaMulticore
import shutil

# 檢查資料夾是否存在，如果不存在就創建資料夾
def create_folder(folder_path):
    print('** def create_folder **')
    try:
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"資料夾 '{folder_path}' 創建成功！")
        else:
            print(f"資料夾 '{folder_path}' 已經存在。")
    except Exception as e:
        print(f"創建資料夾時發生錯誤：{e}")
    
    

# 保存備份raw data
def copy_rawData(source_file_path:str, target_file_path:str):
    print('** def copy_rawData **')
    # 複製單一csv檔案到另一個目錄之中
    try:
        shutil.copy(source_file_path, target_file_path)
        #另一種備份方法:
        #df = pd.read_csv(source_file_path)
        #df.to_csv(target_file_path, index=False)
        print('備份原檔案成功')
    except:
        print('備份原檔案時發生錯誤')

    return




# 簡單記個txt筆記
from datetime import datetime

def create_txt_file(contents:str, txt_file_name:str):
    print('** def create_txt_file**')

    try:
        file_path = txt_file_name + '.txt'
        with open(file_path, 'a+') as file:
            # 獲取當前日期和時間
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 在每次新增內容時換行，並標註寫檔日期和時間
            file.write(f"\n{current_time}\n{contents}")
        print('成功寫入新資訊', contents, '至TXT檔案', txt_file_name)
    except Exception as e:
        print(f"寫TXT檔案時出現錯誤: {e}")





def read_csv_files_in_directory(directory_path:str):  
    print('** def read_csv_files_in_directory**')
    # 檢查目錄是否存在
    if not os.path.isdir(directory_path):
        print(f"目錄 '{directory_path}' 不存在")
        return
    
    # 遍歷目錄中的所有檔案和子目錄
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 確保檔案是 CSV 格式
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"正在讀取檔案: {file_path}")
                
                # 使用 pandas 讀取 CSV 檔案
                try:
                    df = pd.read_csv(file_path)
                    # 在這裡可以對資料進行處理
                   # print(df.head())  # 這裡只是示例，顯示檔案的前幾行
                except Exception as e:
                    print(f"讀取檔案時出現錯誤: {e}")





def get_Col_data(df, featureCols: list)-> dict:
    print('** def get_Col_data **')
    feature_col_dict = {}
    for colName in featureCols:
        try:
            # 使用 loc 函數選取特定列的所有值
            feature_values = df[colName].tolist()
            # 將特徵值打包成列表並添加到字典中
            feature_col_dict.update({colName:feature_values})
        except KeyError:
            print(f"未找到列名 {colName}")
    return feature_col_dict





def Add_Collumn_to_file(New_Collumn_Name:str,  New_Collumn_list:list, csv_file_path:str):
    print('** def Add_Collumn_to_file **')
    
    # 開啟舊的 CSV 檔案
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"找不到檔案: {csv_file_path}")
        return
    except Exception as e:
        print(f"讀取檔案時出現錯誤: {e}")
        return
    df[New_Collumn_Name] = New_Collumn_list

    # 指定您想要保存的檔案路徑
    output_file_path = csv_file_path

# 將更新後的 DataFrame 寫入 CSV 檔案
    try:
        df.to_csv(output_file_path, index=False)
        print("DataFrame 已成功保存到檔案:", output_file_path)
    except Exception as e:
        print("保存 DataFrame 到檔案時出現錯誤:", e)
    






def count_creator_engagement(retweets, comments, likes, followers:int=1) -> float:
    print('** def count_creator_engagement **')
    
#    retweets = strList_To_NumberList(list_with_comma=retweets)
#    comments = strList_To_NumberList(list_with_comma=comments)
#    likes = strList_To_NumberList(list_with_comma=likes)
    
    # 計算創作者參與度
    creator_engagement = (sum(retweets) + sum(comments) + sum(likes)) / followers /len(likes)
    
    return creator_engagement


# In[31]:


def count_engagement_Threshold(directory_path)-> float:
    print('** def count_engagement_Threshold **')
    
    if not os.path.isdir(directory_path):
        print(f"目錄 '{directory_path}' 不存在")
        return
    
    creator_engagement = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"正在讀取檔案: {file_path}")
                
                try:
                    df = pd.read_csv(file_path)
                    creator_engagement.append(df['creator_engagement'].iloc[0]) 
                    
                except Exception as e:
                    print(f"讀取檔案時出現錯誤: {e}")
                    
                    
    creator_engagement_mean = sum(creator_engagement)/len(creator_engagement) #取平均
    print('engagement_Threshold: ',creator_engagement_mean)
    return creator_engagement_mean







def judje_creator_success(creator_engagement:float, engagement_Threshold: float):
    if creator_engagement>= engagement_Threshold:
        return 1
    else:
        return 0






def str_To_Number(value_with_comma:str)->int:
    print('** def str_To_Number **')
    print(f'value_with_comma:{value_with_comma}')
    
    if ',' in value_with_comma:
        value_with_comma = value_with_comma.replace(',', '')
    if 'K' in value_with_comma :
        value_with_comma = value_with_comma.replace('K','000') 
    
    if 'M' in value_with_comma:
        value_with_comma = value_with_comma.replace('M','000000') 

    if '.0' in value_with_comma:
        value_with_comma = value_with_comma.replace('.0', '')
    if '.' in value_with_comma:
        value_with_comma = value_with_comma.replace('.', '')
        value_with_comma = value_with_comma.replace('0', '', 1)
    
    value_without_comma = value_with_comma
    integer_value = int(value_without_comma)
    print(f'integer_value:{integer_value}')
    
    return integer_value








def strList_To_NumberList(list_with_comma:list)->list:
    print('** def strList_To_NumberList **')
    print(f'value_with_comma:{list_with_comma}')
    
    list_without_comma = []
    print('here')
    for value_with_comma in list_with_comma:
        if ',' in value_with_comma:
            value_with_comma = value_with_comma.replace(',', '')
        if 'K' in value_with_comma :
            value_with_comma = value_with_comma.replace('K','000') 
        if 'M' in value_with_comma:
                value_with_comma = value_with_comma.replace('M','000000') 

        if '.0' in value_with_comma:
                value_with_comma = value_with_comma.replace('.0', '')
        if '.' in value_with_comma:
                value_with_comma = value_with_comma.replace('.', '')
                value_with_comma = value_with_comma.replace('0', '', 1)
        
        value_without_comma = value_with_comma    
        integer_value = int(value_without_comma)
        list_without_comma.append(integer_value)
    
    
    print(f'integer_value:{list_without_comma}')
    
    return list_without_comma








def Classify_influencerType(follower:str)-> str:
    print('** def Classify_influencerType **')
    
    follower = str_To_Number(follower)
    influencerType = ''
    
    if follower >= 1000000:
        influencerType = 'MegaInfluencer'
    elif follower < 1000000 and follower>=100000:
        influencerType = 'MacroInfluencer'
    elif follower < 100000 and follower>=1000:
        influencerType = 'MicroInfluencer'
    else:
        influencerType = 'NanoInfluencer'
    
    
    print(f'follower:{follower}, influencerType:{influencerType}')
    
    return influencerType 
    






def detect_urls_hashtags_metions(text:str)-> str:
    #print('** detect_urls_and_hashtags **')

    # 網址的正則表達式
    url_pattern = r'https?://\S+'
    
    # hashtag的正則表達式
    hashtag_pattern = r'#\w+'
    
    # 提及（@）的正則表達式
    mention_pattern = r'@\w+'
    
    urls = re.findall(url_pattern, text)
    
    hashtags = re.findall(hashtag_pattern, text)
    
    mentions = re.findall(mention_pattern, text)
    
    # 移除文字中的網址
    text_removed = re.sub(url_pattern, '', text)
    
    return text_removed, len(urls), len(hashtags), len(mentions) ##########計算提及##########




def detect_emoji(text:str)-> tuple:
    #print('** detect_emoji **')
    
    #urls = re.findall(url_pattern, text)

    text = emoji.demojize(text)
    
    emoji_count = sum(word.startswith(":") and word.endswith(":") for word in text)#############
    
    text_without_emoji = text.replace(":", "").replace("_face", "")


    # print("轉換後的文字:", text_with_names)
    return text_without_emoji, emoji_count







def calculate_monthly_post_stability(df):
    """
    計算每個月的貼文數佔總貼文數的比率以及發文頻率的穩定度

    Parameters:
    df (DataFrame): 包含帖子日期的數據框（DataFrame）

    Returns:
    months (list): 每個月份的列表（字符串格式）
    post_ratios (list): 每個月份貼文數量佔總貼文數的比率（浮點數格式）
    post_stability (float): 發文頻率的穩定度（標準差）
    total_posts (int): 總貼文數
    """
    # 將帖子日期轉換為 datetime 格式
    df['postDate'] = pd.to_datetime(df['postDate'])
    
    # 提取年份和月份，並添加新的欄位
    df['year_month'] = df['postDate'].dt.to_period('M')
    
    # 計算每個月的帖子數量
    monthly_counts = df['year_month'].value_counts().sort_index()
    
    # 計算總帖子數
    total_posts = monthly_counts.sum()
    
    # 計算每個月份的帖子數量佔總帖子數的比率
    post_ratios = monthly_counts / total_posts
    
    # 計算發文頻率的穩定度（標準差）
    post_stability = np.std(post_ratios)
    
    # 將結果轉換為兩個 list
    months = [str(month) for month in post_ratios.index]  # 將月份轉換為字符串格式

    return months, post_ratios.tolist(), post_stability, total_posts







def calculate_text_ratios(textdata):
    print('** def calculate_text_ratios **')

    # 计算使用hashtag的比率
    num_hashtags = textdata['numOfHashtags'].sum()
    total_tweets = textdata.shape[0]
    hashtag_ratio = num_hashtags / total_tweets

    # 计算使用emoji的比率
    num_emojis = textdata['numOfEmojis'].sum()
    emoji_ratio = num_emojis / total_tweets

    num_url = textdata['numOfUrls'].sum()
    url_ratio = num_url / total_tweets
    
    num_mention = textdata['numOfMentions'].sum()
    mention_ratio = num_url / total_tweets

    return hashtag_ratio, emoji_ratio, url_ratio, mention_ratio





# Function to map Penn Treebank POS tag to WordNet POS tag
def get_wordnet_pos(word:str):
    #print('** get_wordnet_pos **')

    # Get POS tag using nltk.pos_tag and map to WordNet POS
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ,  # Adjective
                "N": wn.NOUN,  # Noun
                "V": wn.VERB,  # Verb
                "R": wn.ADV}   # Adverb
    
    return tag_dict.get(tag, wn.NOUN)  # Default to Noun if not found





'''
def text_preprocessing(text:str)->list:
       
    text = text.translate(str.maketrans('', '', string.punctuation))  #刪去標點符號
    text = text.lower() # 統一轉為小寫

    sentences = nltk.sent_tokenize(text) # 斷句 
    tokens = [nltk.tokenize.word_tokenize(sent) for sent in sentences]  # 斷詞
    
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    tokens = [token for token in tokens[0] if token not in nltk_stopwords] # 僅保留非停用字(去除停用字)
        
    return tokens # 需要回傳list

'''



def clean_text(text:str)->str:
    #print('** clean_text **')

    # Remove non-English text
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    #text, url_count, hashtag_count = detect_urls_and_hashtags(text) # remove url and hashtags
    #text = detect_emoji(text) # replace emoji to words
    
    # Remove emojis
    #text = re.sub('[^\w\s,]', '', text)
    
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Lower Case
    text = text.lower()
    
    # Tokenize text into words
    words = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word in words:
        pos = get_wordnet_pos(word)
        if pos:
            lemma = lemmatizer.lemmatize(word, pos)
            lemmatized_words.append(lemma)
    
    # Join words back into text
    cleaned_text = ' '.join(lemmatized_words)
    
    return cleaned_text







# 計算每一篇貼文的情緒 
def Sentiment_Analysis(tweets:list):
    print('** def Sentiment_Analysis **')
    
    polarity = []
    subjectivity=[]
    
    for tweet in tweets:
        blob = TextBlob(tweet)
       # print(blob.sentiment) 
        polarity.append(blob.sentiment.polarity)  # polarity 的值在範圍 [-1, 1]，表示情感的正負程度
        subjectivity.append(blob.sentiment.subjectivity)  #subjectivity 的值在範圍 [0, 1]，表示文本的主觀性。
        print(f'Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}')
        
    return polarity, subjectivity
    






def count_rate_of_pos_Sentiment(df)->float:
    print('** def count_rate_of_pos_Sentimentt **')
    
    numOfTweet = len(df)  # 統計總推文數量
    PositiveTweet = df.loc[df['sentiment'] > 0]  
    numOfPositive = len(PositiveTweet)  # 統計正向推文數量
    rate_of_post_Sentiment = numOfPositive / numOfTweet  # 計算正向推文佔比
    
    print(f'正貼文比: {rate_of_post_Sentiment}')
    return rate_of_post_Sentiment








def count_avg_Subjectivity(textdata):
    
    return textdata['subjectivity'].mean()





def train_LDA_model(documents:list, num_topics:int):
    print('** def train_LDA_model **')

    # 斷詞
    texts = [[word for word in document.split()] for document in documents]

    # 建詞典
    dictionary = corpora.Dictionary(texts)

    # 建詞频矩陣
    corpus = [dictionary.doc2bow(text) for text in texts]

    # train LDA model
    #lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20) #這個速度比較慢
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics)

    #print(lda_model.print_topics())
    return lda_model







def topic_analysis(documents:list, lda_model):
   # print('** def topic_analysis **')
    document_topics = []
    for document in documents:
        # 分词处理
        words = [word for word in document.split()]
        # 将文档转换为词袋表示
        document_bow = lda_model.id2word.doc2bow(words)
        # 获取文档的主题分布
        topics = lda_model.get_document_topics(document_bow)
        # 选择具有最高概率的主题作为分类标签
        if topics:
            dominant_topic = max(topics, key=lambda x: x[1])[0]
        else:
            dominant_topic = None
        document_topics.append(dominant_topic)
    return document_topics









def count_posts_per_topic(topic_tag: list, num_of_topics: int):
    print('** def count_posts_per_topic **')
    
    posts_per_topic = {}
    for topic_index in range(num_of_topics):  
        count = topic_tag.count(topic_index)
        topic_name = f'topic{topic_index+1}'# 主題索引從1開始
        posts_per_topic[topic_name] = count/len(topic_tag)
    df = pd.DataFrame.from_dict(posts_per_topic, orient='index', columns=['Count'])
    
    return df #posts_per_topic 




''' 
def append_num_of_topic_to_csv(posts_per_topic: dict, csv_filename: str):
    print('** def append_num_of_topic_to_csv **')
    
    # 將字典轉換為 DataFrame
    df = pd.DataFrame.from_dict(posts_per_topic, orient='index', columns=['Value'])
    
    # 將 DataFrame 寫入 CSV 文件，並指定 index_label 為索引列
    df.to_csv(csv_filename, mode='a', header=False, index_label='Index')
   
    return
   
append_num_of_topic_to_csv(posts_per_topic=posts_per_topic, csv_filename='text.csv')
'''





import os
import pandas as pd
import numpy as np

def get_topic_and_write(topic_folder: str, user_folder: str):
    # 檢查資料夾是否存在
    if not os.path.exists(topic_folder) or not os.path.exists(user_folder):
        print("指定的資料夾不存在。")
        return
    
    # 遍歷主題資料夾中的所有文件
    for topic_file in os.listdir(topic_folder):
        
        # 讀取主題 CSV 文件
        try:
            topic_path = os.path.join(topic_folder, topic_file)
            topic_df = pd.read_csv(topic_path)
        except Exception as e:
            print("無法讀取主題文件:", e)
            continue
        try:
            user_path = os.path.join(user_folder, topic_file)
            user_path = user_path.replace('.csv','_bg.csv')
            user_df = pd.read_csv(user_path)
            
            
            # 找到主題
            topic_count = topic_df['Count']
            idx = np.where(topic_count == 1)[0]
            topic = topic_count.index[idx[0]]
        
            print(f'-----{topic}-----')
            # 將主題資訊應用到使用者 DataFrame 中
            user_df['creator_topic'] = topic
            
            # 保存修改後的使用者 DataFrame
            user_df.to_csv(user_path, index=False)
            print(f"已成功將主題資訊應用到使用者文件 {user_path} 中並保存。")
        except Exception as e:
            print(f"無法保存使用者文件 {user_path}:", e)






# Function to convert Penn Treebank POS tag to WordNet POS tag
def ptb_to_wn(tag):  
    #print('** def ptb_to_wn **')

    if tag.startswith('N'):
        return 'n'  # Noun
    if tag.startswith('V'):
        return 'v'  # Verb
    if tag.startswith('J'):
        return 'a'  # Adjective
    if tag.startswith('R'):
        return 'r'  # Adverb
    return None  # Return None for other cases

# Function to convert tagged word to WordNet synset
def tagged_to_synset(word, tag):
    #print('** def tagged_to_synset **')
    
    wn_tag = ptb_to_wn(tag)  # Convert Penn Treebank POS tag to WordNet POS tag
    if wn_tag is None:
        return None  # Return None if POS tag is not recognized 
    try:
        # Get the first synset for the word and POS tag
        return wn.synsets(word, wn_tag)[0]
    except:
        return None  # Return None if no synsets are found
    





# Function to calculate the similarity score between two sentences
def sentence_similarity(s1, s2):    
    #print('** def sentence_similarity **')
    
    # Tokenize and POS tag the input sentences
    s1 = pos_tag(word_tokenize(s1))
    s2 = pos_tag(word_tokenize(s2)) 

    # Convert POS-tagged words to WordNet synsets
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in s1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in s2]

    # Remove "None" values from synsets
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # Calculate the similarity score for each synset in the first sentence
    for synset in synsets1:
        try:
            # Find the best similarity score with synsets in the second sentence
            best_score = max([synset.path_similarity(ss) for ss in synsets2])
            
            if best_score is not None:
                score += best_score
                count += 1
        except:
            score = 0  # Handle exceptions by setting score to 0

    try:
        score /= count  # Calculate the average score
    except ZeroDivisionError:
        score = 0  # Set score to 0 if there are no valid scores

    return score

# Function to compute symmetric sentence similarity
def symSentSim(s1, s2):
    #print('** def symSentSim **')
    # Calculate the symmetric sentence similarity score
    sss_score = (sentence_similarity(s1, s2) + sentence_similarity(s2, s1)) / 2
    return sss_score



def personality_analysis(df, file_path:str):
    print('** def personality_analysis **')
    
    try:
        extraversion = ['talkative', 'assertive', 'enthusiasm', 'energetic', 'adventure', 'dominance', 'social', 'excitement', 'fun', 'optimism']
        agreeableness = ['politeness', 'helpful', 'kind', 'empathy', 'cooperation', 'modesty', 'affection', 'sympathy', 'pleasant', 'trust']
        conscientiousness = ['achievement', 'striving', 'planning', 'organized', 'dutiful', 'discipline', 'work', 'responsible', 'dependable', 'perseverance']
        neuroticism = ['anxiety', 'anger', 'depression', 'emotional', 'stress', 'vulnerability', 'fear', 'nervous', 'tense', 'worry']
        openness = ['insight', 'curious', 'interest', 'imagination', 'unconventional', 'originality', 'creativity', 'art', 'novel', 'idea']
        o = ' '.join(openness)
        c = ' '.join(conscientiousness)
        e = ' '.join(extraversion)
        a = ' '.join(agreeableness)
        n = ' '.join(neuroticism)


        #df['CleanedText'] = df['textWithoutEmoji'].apply(clean_text)

        # 比對詞彙意義相似度
        df['O_Score'] = df['CleanedText'].apply(lambda x: symSentSim(x, o)) 
        df['C_Score'] = df['CleanedText'].apply(lambda x: symSentSim(x, c))
        df['E_Score'] = df['CleanedText'].apply(lambda x: symSentSim(x, e))
        df['A_Score'] = df['CleanedText'].apply(lambda x: symSentSim(x, a))
        df['N_Score'] = df['CleanedText'].apply(lambda x: symSentSim(x, n))
        print('Get OCEAN Score success!')
    except Exception as e:
        print("計算大五人格分數時出現錯誤:", e)
    
    try:
        big5_file_path = file_path.replace('.csv','_big5.csv')#save(new file)
        df.to_csv(big5_file_path, index=False)
    except Exception as e:
        print("無法正確保存大五人格分數:", e)







def calculate_ocean_avg_scores(user_file_path: str, text_file_path:str):
    print('** def calculate_ocean_avg_scores **')

    try:
        # 从文件中读取数据
        userFile = pd.read_csv(user_file_path)
        textFile = pd.read_csv(text_file_path)
        
        # 计算每个维度的平均分数
        userFile['O_Score'] = [textFile['O_Score'].mean()]
        userFile['C_Score'] = [textFile['C_Score'].mean()]
        userFile['E_Score'] = [textFile['E_Score'].mean()]
        userFile['A_Score'] = [textFile['A_Score'].mean()]
        userFile['N_Score'] = [textFile['N_Score'].mean()]
            
    except Exception as e:
        print("计算 OCEAN 平均分数时出现错误:", e)

    try:
        userFile.to_csv(user_file_path, index=False)
        
    except Exception as e:
        print("存檔错误:", e)







def calculate_follower_ocean_avg_scores(user_file_path: str, text_file_path:str):
    print('** def calculate_ocean_avg_scores **')

    try:
        # 从文件中读取数据
        userFile = pd.read_csv(user_file_path)
        textFile = pd.read_csv(text_file_path)
        
        # 计算每个维度的平均分数
        userFile['Follower_O_Score'] = [textFile['O_Score'].mean()]
        userFile['Follower_C_Score'] = [textFile['C_Score'].mean()]
        userFile['Follower_E_Score'] = [textFile['E_Score'].mean()]
        userFile['Follower_A_Score'] = [textFile['A_Score'].mean()]
        userFile['Follower_N_Score'] = [textFile['N_Score'].mean()]
        print(textFile['N_Score'].mean())
    except Exception as e:
        print("计算 OCEAN 平均分数时出现错误:", e)

    try:
        userFile.to_csv(user_file_path, index=False)
        
    except Exception as e:
        print("存檔错误:", e)



