# Create your views here.
from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel, EncryptionModels
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np
import random
import os


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

def top_directors(request):
    path=os.path.join(settings.MEDIA_ROOT,'movie_success_rate.csv')
    imdb_data = pd.read_csv(path)
    imdb_data.Director.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))
    plt.title('TOP 10 DIRECTORS OF MOVIES')
    plt.show()
    return redirect(UserHome)

path=os.path.join(settings.MEDIA_ROOT,'movie_success_rate.csv')
imdb_data = pd.read_csv(path)
imdb_data=imdb_data.rename(columns = {'Revenue (Millions)':'Revenue_Millions'})
imdb_data=imdb_data.rename(columns = {'Runtime (Minutes)':'Runtime_Minutes'})

# warnings.filterwarnings("ignore")
# imdb_data = imdb_data.drop("Genre", axis = 1)
imdb_data=imdb_data[['Title','Director','Runtime_Minutes','Rating','Revenue_Millions','Action','Adventure','Comedy','Family','Horror','Sport','Success']]
imd = imdb_data
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
imd['Title']=lb.fit_transform(imd['Title'])
imd['Director']=lb.fit_transform(imd['Director'])
x = imd[imd.columns[0:11]]
y = imd["Success"]
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.2)

def alg(request):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix,accuracy_score
    from sklearn.metrics import ConfusionMatrixDisplay
    sns.stripplot(x="Revenue_Millions", y="Rating", data=imdb_data, jitter=True);
    plt.title(' RATING BASED ON YEAR')
    plt.show
    imd["Rating"].value_counts() 
    Sortedrating= imdb_data.sort_values(['Rating'], ascending=False)
    mediumratedmovies= imdb_data.query('(Rating > 3.0) & (Rating < 7.0)')
    highratedmovies= imdb_data.query('(Rating > 7.0) & (Rating < 10.0)')
    sns.jointplot(x="Rating", y="Revenue_Millions", data=highratedmovies);
    plt.title('(MOVIES WITH HIGH RATING ,REVENUE')
    metascore=imd.Rating
    sns.boxplot(metascore)
    plt.show()

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred_rf_test = rf.predict(x_test)
    y_pred_rf_train = rf.predict(x_train)
    testacc_rf = accuracy_score(y_pred_rf_test, y_test)
    trainacc_rf = accuracy_score(y_pred_rf_train, y_train)
    print("Training Accuracy of Random Forest: ", trainacc_rf)
    print("Testing Accuracy of Random Forest: ", testacc_rf)
    sns.heatmap(confusion_matrix(y_test,y_pred_rf_test), fmt = 'd',annot=True, cmap='magma')
    print(classification_report(y_test,y_pred_rf_test))
    cm=confusion_matrix(y_pred_rf_test,y_test)
    print(cm)
    import matplotlib.pyplot as plt
    confusion=ConfusionMatrixDisplay(confusion_matrix=cm)
    confusion.plot()
    plt.title('Cofusion matrix for RandomForestClassifier')
    plt.show()

    import xgboost
    xg = xgboost.XGBClassifier()
    xg.fit(x_train,y_train)
    y_pred = xg.predict(x_test)
    b=accuracy_score(y_pred,y_test)
    print('Cofusion matrix for XGboost')
    print('accuracy score is',b)
    cm=confusion_matrix(y_pred,y_test)
    print(cm)
    import matplotlib.pyplot as plt
    confusion=ConfusionMatrixDisplay(confusion_matrix=cm)
    confusion.plot()
    plt.title('Cofusion matrix for XGboost')
    plt.show()

    from sklearn.svm import SVC
    rf=SVC()
    rf.fit(x_train,y_train)
    y_pre=rf.predict(x_test)
    a=accuracy_score(y_pre,y_test)
    cm=confusion_matrix(y_pre,y_test)
    print(cm)
    print(a)
    confusion_matrix=ConfusionMatrixDisplay(confusion_matrix=cm)
    import matplotlib.pyplot as plt
    confusion_matrix.plot()
    plt.title('Confusion Matrix for SVM')
    plt.show()

    from sklearn.neighbors import KNeighborsClassifier 
    from sklearn.metrics import ConfusionMatrixDisplay   
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix,accuracy_score
    # Assuming you have x_train, x_test, y_train, and y_test defined
    # Create and train the KNN model
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    # Make predictions on the test set
    y_pred_knn = knn.predict(x_test)
    # Evaluate the model
    e = accuracy_score(y_pred_knn, y_test)
    print('Accuracy score for KNN:', e)
    cm_knn = confusion_matrix(y_pred_knn, y_test)
    print('Confusion Matrix for KNN:')
    print(cm_knn)
    # Plot Confusion Matrix
    confusion_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
    confusion_knn.plot()
    plt.title('Confusion Matrix for KNN')
    plt.show()


    return render(request,'users/alg.html',{'testacc_rf':testacc_rf,'a':a,'b':b,'e':e})



# Initialize SimpleImputer with the desired strategy (e.g., mean)
imputer = SimpleImputer(strategy='mean')

def prediction(request):
    if request.method == 'POST':
        # Encode categorical variables
        Title = lb.fit_transform([request.POST.get('title')])
        Director = lb.fit_transform([request.POST.get('director')])

        # Handle missing values using SimpleImputer
        Runtime_Minutes = float(request.POST.get('runtime'))
        Rating = float(request.POST.get('rating'))
        Revenue_Millions = float(request.POST.get('revenue'))
        Action = int(request.POST.get('action'))
        Adventure = float(request.POST.get('adventure'))  # Assuming Adventure is a numeric feature
        Comedy = int(request.POST.get('comedy'))
        Family = int(request.POST.get('family'))
        Horror = int(request.POST.get('horror'))
        Sport = int(request.POST.get('sport'))

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Title': Title,
            'Director': Director,
            'Runtime_Minutes': Runtime_Minutes,
            'Rating': Rating,
            'Revenue_Millions': Revenue_Millions,
            'Action': Action,
            'Adventure': Adventure,
            'Comedy': Comedy,
            'Family': Family,
            'Horror': Horror,
            'Sport': Sport
        })

        # Fit the SimpleImputer on the training data
        imputer.fit(x_train)

        # Impute missing values
        input_data_imputed = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)

        # Initialize HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier()

        # Assuming 'x_train' contains your features and 'y_train' contains your labels
        clf.fit(x_train, y_train)

        # Make a prediction
        value = clf.predict(input_data_imputed)
        if value==1:
            value='Success'
        else:
            value='Fail'

        return render(request, 'users/result.html', {'value': value})

    return render(request, 'users/prediction.html', {})