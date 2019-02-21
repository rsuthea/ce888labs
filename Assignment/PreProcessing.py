##skeleton program for cleaning all datasets
def CleanStudentData(student_df):
    #clean catagorical data using one-hot encoding
    #this should be done with:
    #school
    #sex
    #address (change to boolean value)
    #family size(change to boolean)
    #mother/father job
    #travel time
    #study time
    return cleaned_StudentData

def CleanEmoji(emoji_df):
    for x in emoji_df:
        #create an array with the frequency of terms based on the emoji used
        BagOfWords(x)
        sentimentOfText = senitimentTree(x)
        #create an array with the frequency of sentiment vased on the emoji used
        BagOfWords(sentimentOfText)
    return cleaned_Emoji

def CleanBC(bc_df):
    for x in bc_df:
        if x[1] =="m":
            x[1]=1
        else:
            x[1]=0
    return cleaned_BC

#these are purely the url's for the datasets, not the datasets themselves
student_df = "https://archive.ics.uci.edu/ml/datasets/student+performance"
emoji_df = "https://www.kaggle.com/hariharasudhanas/twitter-emoji-prediction"

bc_df = "https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(Diagnostic)"

a = CleanStudentData(student_df)
b = CleanEmoji(emoji_df)
c = CleanBC(bc_df)