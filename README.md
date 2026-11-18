# SMS Spam Detection Using Machine Learning

A machine learning project that classifies SMS messages as **Ham (Legitimate)** or **Spam (Unwanted)** using Natural Language Processing (NLP) and supervised machine learning algorithms.

The project compares different text vectorization techniques and classification models to identify an effective approach for SMS spam detection.

---

## 1. Project Overview

SMS spam messages can contain fraudulent offers, unwanted advertisements, suspicious links, and misleading prize notifications. Manually identifying these messages can be difficult.

The objective of this project is to develop a machine learning system that automatically identifies whether an SMS message is legitimate or spam.

We experiment with three model configurations:

1. Multinomial Naive Bayes + CountVectorizer
2. Multinomial Naive Bayes + TF-IDF
3. Logistic Regression + TF-IDF

The models are evaluated using accuracy, precision, recall, F1-score, and stratified 5-fold cross-validation.

## 2. Dataset

**Dataset:** SMS Spam Collection Dataset  
**Source:** UCI Machine Learning Repository

Dataset link: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains 5,572 SMS messages with two categories:

| Label | Meaning |
|---|---|
| Ham | Legitimate SMS messages |
| Spam | Unwanted or fraudulent SMS messages |

The original dataset contains two main fields:

- `label`: The category of the message (`ham` or `spam`).
- `message`: The actual SMS text.

### Target Label Encoding

We convert the categorical labels into numerical values:

| Original Label | Encoded Value |
|---|---:|
| Ham | 0 |
| Spam | 1 |

This allows the machine learning algorithms to work with numerical target labels.

## 3. Project Workflow

The project follows these major stages:

1. Import required libraries.
2. Load the dataset.
3. Explore and understand the dataset.
4. Encode the target labels.
5. Preprocess the SMS messages.
6. Split the dataset into training and testing sets.
7. Convert text into numerical features.
8. Train different machine learning models.
9. Evaluate training and testing performance.
10. Perform stratified 5-fold cross-validation.
11. Compare the models.
12. Test the trained models using new SMS messages.

---

## 4. Import Required Libraries

We use Python libraries for data processing, text preprocessing, visualization, and machine learning.

| Library | Purpose |
|---|---|
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Regular Expressions (`re`) | Text preprocessing |
| NLTK | Natural Language Processing utilities |
| Scikit-learn | Feature extraction, model training, and evaluation |

The main machine learning components come from Scikit-learn:

- `CountVectorizer`
- `TfidfVectorizer`
- `MultinomialNB`
- `LogisticRegression`
- `train_test_split`
- `StratifiedKFold`
- `Pipeline`
- Evaluation metrics and classification reports

## 5. Load and Explore the Dataset

First, we load the dataset into a Pandas DataFrame.

We then examine:

- The first few records.
- The dataset shape.
- Column names and data types.
- Missing values.
- The distribution of Ham and Spam messages.

### Why Explore the Dataset?

Exploratory Data Analysis (EDA) helps us understand the structure and characteristics of the data before training a model.

We also examine the class distribution to understand whether the dataset is balanced.

The SMS Spam Collection contains significantly more legitimate messages than spam messages, so class imbalance must be considered when evaluating model performance.

## 6. Target Label Encoding

Machine learning classifiers require numerical target labels.

We convert:

- `ham` → `0`
- `spam` → `1`

This makes it possible for the models to learn the relationship between message features and their corresponding categories.

## 7. Text Preprocessing

Raw SMS messages contain uppercase and lowercase letters, punctuation, numbers, phone numbers, URLs, currency symbols, and other elements.

Before feature extraction, we clean and normalize the text.

### Initial Preprocessing Approach

Initially, we experimented with the following operations:

1. Remove non-alphabetic characters.
2. Convert text to lowercase.
3. Split messages into individual words.
4. Remove English stopwords.
5. Apply Porter Stemming.
6. Join the words back into a cleaned sentence.

However, this approach removes potentially useful spam indicators.

For example, words related to phone numbers, money, URLs, and percentages can provide important information for spam classification.

### Improved Preprocessing Approach

To preserve these indicators, we introduced a more suitable preprocessing function.

The improved approach performs the following operations:

**Step 1: Convert text to lowercase**

Makes the text case-insensitive.

Example:

`CONGRATULATIONS` → `congratulations`

**Step 2: Identify URLs**

URLs are replaced with a common token.

`www.example.com` → `urltoken`

**Step 3: Identify email addresses**

Email addresses are replaced with:

`emailtoken`

**Step 4: Identify currency amounts**

Currency values are replaced with:

`moneytoken`

Example:

`£5000` → `moneytoken`

**Step 5: Identify phone numbers**

Phone numbers are replaced with:

`phonetoken`

**Step 6: Identify percentages**

Percentage values are replaced with:

`percenttoken`

**Step 7: Normalize remaining numbers**

Other numerical values are replaced with:

`numbertoken`

**Step 8: Remove remaining non-alphabetic characters**

Finally, we remove non-alphabetic characters and normalize whitespace.

### Improved Preprocessing Function

```python
def preprocess_text(text):
    text = str(text).lower()

    text = re.sub(
        r'(https?://\S+|www\.\S+)',
        ' urltoken ',
        text
    )

    text = re.sub(
        r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b',
        ' emailtoken ',
        text
    )

    text = re.sub(
        r'[£$€]\s*\d+(?:[.,]\d+)?',
        ' moneytoken ',
        text
    )

    text = re.sub(
        r'(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)',
        ' phonetoken ',
        text
    )

    text = re.sub(
        r'\b\d+(?:[.,]\d+)?\s*%',
        ' percenttoken ',
        text
    )

    text = re.sub(
        r'\b\d+(?:[.,]\d+)?\b',
        ' numbertoken ',
        text
    )

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    return ' '.join(text.split())
```

### Example

Original message:

`Congratulations! You won £5000. Call 08001234567 now!`

After preprocessing:

`congratulations you won moneytoken call phonetoken now`

### Why Did We Improve Preprocessing?

The improved approach preserves potentially useful information about the structure of an SMS message.

For example:

- `moneytoken` may indicate a financial offer or prize.
- `phonetoken` may indicate a number to call.
- `urltoken` may indicate an external link.
- `percenttoken` may indicate a promotional offer.
- `emailtoken` may indicate an email contact.

We intentionally do not remove stopwords or apply stemming in this improved version. This allows us to retain more contextual information from the original messages.

---

## 8. Train-Test Split

After preprocessing, we divide the dataset into two parts:

- **80% Training Data:** Used to train the machine learning models.
- **20% Testing Data:** Used to evaluate performance on previously unseen messages.

We use:

```python
train_test_split(
    corpus,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
```

### Why These Parameters?

| Parameter | Purpose |
|---|---|
| `test_size=0.2` | Reserves 20% of the dataset for testing |
| `random_state=42` | Ensures reproducibility |
| `stratify=Y` | Preserves the original class distribution in both sets |

The resulting test set contains 1,115 messages:

- Ham: 966
- Spam: 149

### Important: Preventing Data Leakage

We fit the text vectorizers only on the training data.

We do not use `fit_transform()` on the entire dataset before splitting.

This prevents information from the testing data from influencing the vocabulary learned during training.

---

## 9. Feature Extraction

Machine learning algorithms cannot directly understand raw text. Therefore, we convert SMS messages into numerical feature vectors.

We experiment with two feature extraction techniques.

### 9.1 CountVectorizer

CountVectorizer converts text into numerical features based on word occurrence counts.

For example:

| Message | free | win | meeting |
|---|---:|---:|---:|
| Free entry to win | 1 | 1 | 0 |
| Meeting tomorrow | 0 | 0 | 1 |

We configure it with:

```python
CountVectorizer(max_features=2500)
```

The `max_features=2500` parameter limits the vocabulary to a maximum of 2,500 features.

### 9.2 TF-IDF Vectorizer

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It assigns numerical weights to words based on their frequency within a message and how common they are across the collection of messages.

The purpose is to give relatively more importance to words that are informative rather than simply frequent.

We configure it with:

```python
TfidfVectorizer(max_features=2500)
```

### Why Compare Both?

CountVectorizer focuses on word occurrence counts, while TF-IDF uses weighted word representations.

Comparing them helps us understand how different text representations affect classification performance.

---

## 10. Machine Learning Models

We train three different model configurations.

### 10.1 Multinomial Naive Bayes + CountVectorizer

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification.

It estimates the probability of a message belonging to a particular class based on the features present in that message.

Configuration:

```python
MultinomialNB()
```

Input features:

`CountVectorizer`

### 10.2 Multinomial Naive Bayes + TF-IDF

In this experiment, we use the same Naive Bayes classifier but change the feature extraction technique.

Configuration:

```python
MultinomialNB()
```

Input features:

`TfidfVectorizer`

This experiment helps us compare frequency-based features against TF-IDF-weighted features for Naive Bayes.

### 10.3 Logistic Regression + TF-IDF

Logistic Regression is a supervised classification algorithm that estimates the probability of a sample belonging to a particular class.

We configure it as follows:

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
```

**Explanation:**

- `class_weight='balanced'`: Adjusts class weights to account for class imbalance.
- `max_iter=1000`: Allows sufficient iterations for the optimization process.
- `random_state=42`: Supports reproducibility where randomness is involved.

Input features:

`TfidfVectorizer`

---

## 11. Model Evaluation

After training, we evaluate the models using several classification metrics.

### 11.1 Accuracy

Accuracy measures the proportion of correctly classified messages.

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

### 11.2 Precision

Precision measures how many messages predicted as spam are actually spam.

\[
Precision = \frac{TP}{TP+FP}
\]

Higher spam precision means fewer legitimate messages are incorrectly classified as spam.

### 11.3 Recall

Recall measures how many actual spam messages are successfully detected.

\[
Recall = \frac{TP}{TP+FN}
\]

Higher spam recall means fewer spam messages are missed.

### 11.4 F1-Score

F1-score is the harmonic mean of precision and recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

It provides a balanced measure of precision and recall.

### 11.5 Confusion Matrix

The confusion matrix helps us understand the types of classification errors.

| | Predicted Ham | Predicted Spam |
|---|---|---|
| Actual Ham | True Negative (TN) | False Positive (FP) |
| Actual Spam | False Negative (FN) | True Positive (TP) |

For spam detection:

- False Positive: A legitimate message is incorrectly classified as spam.
- False Negative: A spam message is incorrectly classified as legitimate.

Both types of errors matter, but their impact may differ depending on the application.

---

## 12. Training vs Testing Performance

We evaluate the models on both the training and testing datasets to understand how well they learn and generalize.

The reported detailed classification results after improved preprocessing are:

### Naive Bayes

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Ham | 0.99 | 0.99 | 0.99 | 966 |
| Spam | 0.97 | 0.94 | 0.95 | 149 |

**Accuracy: approximately 0.99**

### Logistic Regression

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Ham | 0.99 | 0.99 | 0.99 | 966 |
| Spam | 0.94 | 0.95 | 0.95 | 149 |

**Accuracy: approximately 0.99**

*Note: These are the rounded class-wise results from the reported classification outputs. The exact accuracy and aggregate metrics should be taken from the notebook's unrounded output.*

---

## 13. Model Comparison Using TF-IDF

To compare the two classification algorithms under the same TF-IDF feature representation, we evaluate:

- Multinomial Naive Bayes + TF-IDF
- Logistic Regression + TF-IDF

Reported test-set results:

| Metric | Naive Bayes (TF-IDF) | Logistic Regression (TF-IDF) |
|---|---:|---:|
| Accuracy | 0.9767 | 0.9857 |
| Precision | 1.0000 | 0.9404 |
| Recall | 0.8255 | 0.9530 |
| F1-Score | 0.9044 | 0.9467 |

### Interpretation

**Naive Bayes + TF-IDF**

- Achieves perfect reported precision.
- Has lower recall.
- Misses more actual spam messages than Logistic Regression in this experiment.

**Logistic Regression + TF-IDF**

- Achieves higher accuracy.
- Provides higher recall.
- Achieves a higher F1-score.

Therefore, Logistic Regression performs better overall than Naive Bayes in this particular TF-IDF test-set comparison.

---

## 14. Stratified 5-Fold Cross-Validation

A single train-test split provides only one evaluation result. To obtain a more reliable estimate of model performance, we also perform stratified 5-fold cross-validation.

### What Is Cross-Validation?

In 5-fold cross-validation, the training dataset is divided into five approximately equal parts.

The process works as follows:

1. Divide the training dataset into five folds.
2. Train the model using four folds.
3. Evaluate it on the remaining fold.
4. Repeat until every fold has been used for validation.
5. Calculate the average evaluation metrics.

We use:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### Why StratifiedKFold?

It preserves the approximate proportion of Ham and Spam messages in each fold.

This is particularly useful because the dataset contains more legitimate messages than spam messages.

### Preventing Cross-Validation Leakage

We use Scikit-learn pipelines so that the vectorizer is fitted independently within each training fold.

This ensures that validation messages do not influence vocabulary fitting or feature extraction.

### Cross-Validation Results

Reported results after improved preprocessing:

| Model | Accuracy | Precision | Recall | F1-Score | Accuracy Std | F1 Std |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes (Count) | 0.9843 | 0.9374 | 0.9465 | 0.9418 | 0.0020 | 0.0071 |
| Naive Bayes (TF-IDF) | 0.9787 | 1.0000 | 0.8412 | 0.9136 | 0.0026 | 0.0111 |
| Logistic Regression (TF-IDF) | 0.9812 | 0.9157 | 0.9482 | 0.9312 | 0.0032 | 0.0103 |

### Cross-Validation Interpretation

**Naive Bayes + CountVectorizer**

- Achieves the highest average accuracy.
- Achieves the highest average F1-score.
- Has the lowest reported F1-score standard deviation.
- Provides a strong overall balance between precision and recall.

**Naive Bayes + TF-IDF**

- Achieves perfect average precision in the reported folds.
- Has comparatively lower recall.
- Its conservative predictions result in more missed spam messages.

**Logistic Regression + TF-IDF**

- Achieves the highest average recall.
- Provides a competitive F1-score.
- Detects a larger proportion of actual spam messages on average.

### Cross-Validation Conclusion

Based on the reported 5-fold cross-validation results, **Naive Bayes with CountVectorizer is the strongest overall configuration** because it achieves the highest average accuracy and F1-score.

Logistic Regression with TF-IDF remains a strong alternative, particularly when detecting a larger proportion of spam messages is a priority.

---

## 15. Testing with New SMS Messages

Finally, we test the trained models using manually selected SMS messages that are separate from the original training and testing datasets.

The examples include:

- Ordinary personal conversations.
- Promotional messages.
- Prize notifications.
- Messages containing phone numbers.
- Messages containing financial offers.
- Tricky messages containing words such as `free`, `win`, `call`, `urgent`, and `prize` in legitimate contexts.

### Why Use These Messages?

A message containing the word `free` or `win` is not necessarily spam.

For example:

- `Free entry to win a prize!` may be spam.
- `Free time this weekend, want to hang out?` may be legitimate.
- `You have won my heart.` is likely a personal message.

These examples help us observe how the models behave when spam-related words appear in different contexts.

### Prediction Workflow

For a new SMS message:

1. Apply the same preprocessing function used during training.
2. Convert the cleaned text using the corresponding fitted vectorizer.
3. Pass the numerical features to the trained classifier.
4. Predict the message category.
5. Retrieve the predicted-class probability.
6. Display the prediction and confidence percentage.

### Important Consistency Rule

The same preprocessing method must be used during:

- Model training.
- Testing.
- Cross-validation.
- Prediction on new messages.

Each classifier must also use its corresponding fitted vectorizer.

| Model | Vectorizer |
|---|---|
| Naive Bayes (Count) | CountVectorizer |
| Naive Bayes (TF-IDF) | TF-IDF |
| Logistic Regression (TF-IDF) | TF-IDF |

The prediction probability represents the model's estimated probability for its predicted class. It should not be interpreted as a guarantee of correctness.

---

## 16. Final Project Findings

The experiments demonstrate that both feature extraction and classification algorithms influence SMS spam detection performance.

The main findings are:

1. Improved preprocessing provides a way to retain useful spam-related indicators that basic text cleaning can remove.
2. CountVectorizer performs particularly well with Multinomial Naive Bayes in the reported cross-validation experiment.
3. TF-IDF combined with Logistic Regression provides strong spam recall.
4. Naive Bayes with TF-IDF achieves very high precision but comparatively lower recall.
5. Stratified cross-validation provides additional evidence about model performance and consistency.
6. Using the same preprocessing and fitted vectorizer during inference is essential for reliable predictions.

### Selected Model

**Naive Bayes + CountVectorizer** is selected as the strongest overall configuration based on the reported cross-validation accuracy and F1-score.

**Alternative:** Logistic Regression + TF-IDF is a strong alternative when higher spam recall is a priority.

The final choice depends on the application's requirements and the relative cost of false positives and false negatives.

---

## 17. Technologies Used

| Technology | Application |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| NLTK | NLP utilities |
| Scikit-learn | Feature extraction and machine learning |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook / Google Colab | Development environment |

---

## 18. How to Run the Project

### Step 1: Obtain the Dataset

Download the SMS Spam Collection dataset from the UCI Machine Learning Repository.

### Step 2: Open the Notebook

Open the project notebook using Jupyter Notebook or Google Colab.

### Step 3: Install Dependencies

Install the required libraries if they are not already available:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

### Step 4: Load the Dataset

Place the dataset in the appropriate location and update the file path in the notebook if necessary.

### Step 5: Execute the Notebook

Run the notebook cells in order:

1. Library imports.
2. Dataset loading.
3. Exploratory data analysis.
4. Label encoding.
5. Text preprocessing.
6. Train-test splitting.
7. Feature extraction.
8. Model training.
9. Model evaluation.
10. Cross-validation.
11. Model comparison.
12. New-message testing.

### Step 6: Observe the Results

Review the classification reports, confusion matrices, comparison tables, cross-validation results, and predictions for new SMS messages.

---

## 19. Limitations and Future Improvements

Although the models achieve strong results on the selected dataset, several limitations remain.

### Current Limitations

- The project uses a single SMS dataset.
- Performance on messages from other sources or newer spam campaigns may differ.
- Manually selected test messages do not represent every possible SMS pattern.
- The predicted probability is not necessarily a perfectly calibrated confidence score.
- The regex-based preprocessing rules may not recognize every possible URL, phone number, email address, or financial pattern.

### Future Improvements

Possible extensions include:

- Testing on additional SMS datasets.
- Experimenting with n-grams.
- Hyperparameter tuning.
- Probability calibration.
- Threshold optimization based on false-positive and false-negative costs.
- Comparing additional classifiers such as Support Vector Machines and Random Forest.
- Developing a simple web application or API for real-time SMS classification.

---

## 20. Conclusion

This project demonstrates how Natural Language Processing and supervised machine learning can be combined to identify spam SMS messages.

Starting from raw text data, we performed exploratory analysis, preprocessing, feature extraction, model training, and evaluation.

We compared Multinomial Naive Bayes and Logistic Regression using CountVectorizer and TF-IDF representations. We also used stratified 5-fold cross-validation to assess the consistency of the models.

Based on the reported experimental results, **Multinomial Naive Bayes with CountVectorizer achieved the strongest overall cross-validation performance**, while Logistic Regression with TF-IDF demonstrated strong spam detection recall.

The project provides a practical foundation for understanding text classification, feature extraction, model evaluation, and the application of machine learning to spam detection.

---

**Project:** SMS Spam Detection Using Machine Learning  
**Dataset:** UCI SMS Spam Collection  
**Task:** Binary Text Classification  
**Classes:** Ham (0) and Spam (1)  
**Primary Models:** Multinomial Naive Bayes and Logistic Regression  
**Feature Extraction:** CountVectorizer and TF-IDF  
**Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Stratified 5-Fold Cross-Validation
