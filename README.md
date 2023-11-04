# Term Deposit Marketing

### <b>Background</b>

We are a small startup focusing mainly on providing machine learning solutions in the European banking market. We work on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.

We are interested in developing a robust machine learning system that leverages information coming from call center data.

Ultimately, we are looking for ways to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.

### <b>Data Description</b>

The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

#### Attributes:
- age : age of customer (numeric)
- job : type of job (categorical)
- marital : marital status (categorical)
- education (categorical)
- default: has credit in default? (binary)
- balance: average yearly balance, in euros (numeric)
- housing: has a housing loan? (binary)
- loan: has personal loan? (binary)
- contact: contact communication type (categorical)
- day: last contact day of the month (numeric)
- month: last contact month of year (categorical)
- duration: last contact duration, in seconds (numeric)
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

#### Output (desired target):
- y - has the client subscribed to a term deposit? (binary)

### <b>Objectives</b>
- Predict if the customer will subscribe (yes/no) to a term deposit (variable y).
- Hit 81% or above accuracy by evaluating with 5-fold cross validation and reporting the average performance score.
- We are also interested in finding customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.
- What makes the customers buy? Tell us which feature we should be focusing more on.

### <b> Results</b>

<u>Model Performance</u>

The goal was to achieve 81% or higher accuracy in predicting customer subscription (yes/no) to a term deposit (variable y). Utilizing the XGBClassifier model, known for its speed and efficacy, we surpassed the goal with an exceptional average accuracy score of 93.5% through 5-fold cross validation. This high accuracy can be attributed to:

- Abundant data: Leveraging a dataset of 40,000 customers with diverse information.
- Comprehensive Feature Expansion: Skillful transformation of categorical variables into a wider format, expanding from the original 14 to 74 columns (including numerical features) for robust modeling.
- Effective Sampling Strategy: Addressing imbalanced target class by implementing an effective sampling approach.

<u>Customer Segmentations</u>

In order to identify customer segments with a higher likelihood of purchasing the products, we conducted an extensive analysis of customer features using various methodologies. This included a comparative analysis of proportions between two key groups: subscribers and non-subscribers. The analysis yielded valuable insights, summarized as follows:

- Age: While the majority of customers were below 60 years old, the subscription success rate was notably higher for customers over 60 (39%) compared to younger age groups (7%). Consequently, we recommend a strategic focus on the older demographic, as they display a greater propensity to subscribe.

- Housing Loan: Nearly half of the subscribers (49%) did not possess a housing loan, while only 39% of non-subscribers were without a housing loan. This indicates a higher likelihood of subscription for customers without such financial obligations.

- Marital Status: Proportionally, fewer married customers subscribed, whereas single and divorced customers collectively accounted for nearly half of all subscribers. Tailoring offerings to these specific marital status groups could be a strategic approach.

- Education Level: Subscribers, on the whole, have a higher level of education compared to non-subscribers. This suggests an opportunity to design services that cater to less educated individuals or to enhance offerings targeting the more educated demographic. Notably, customers with tertiary education exhibited the highest likelihood to subscribe, with a 9% point difference compared to non-subscribers (36% vs. 27%).

- Job Type: Analysis revealed a significant gap between subscribers and non-subscribers in the blue-collar customer segment. Blue-collar customers comprised 24% of non-subscribers, contrasting with their representation of only 18% among subscribers, indicating a lower likelihood to subscribe for this group. Conversely, customers in management positions demonstrated a higher subscription rate (23% of subscribers vs. 20% of non-subscribers), suggesting a strategic focus on this demographic. Additionally, it's noteworthy that students, retired individuals, and the unemployed exhibited the highest subscriber-to-non-subscriber ratios at 15.6%, 10.5%, and 8.7%, respectively.

Leveraging these insights, our client can tailor marketing strategies to effectively target older age groups, customers without housing loans, single and divorced individuals, those with tertiary education, and those in management positions. This targeted approach is expected to optimize subscription rates and enhance overall marketing effectiveness.

<u>Critical Features</u>

In our quest to determine the factors influencing customer decisions to purchase term deposit products, we delved into various customer features. This exploration aims to guide our client in channeling efforts towards the most impactful aspects.

Based on feature importance scores extracted from the XGBoost classifier, the following features were identified as most significant:

- 'month'
- 'housing'
- 'marital'
- 'contact'
- 'education'
- 'job'
- 'day'

Features other than these demonstrated importance scores below 0.05, indicating their relatively weaker role in predicting customer subscriptions.

While 'day' and 'month' emerged as top features, understanding their correlation with marketing initiatives, as well as taking into account specific company/market situations on certain days, months, and seasons, is imperative. To achieve this understanding, a more extensive dataset encompassing customer subscriptions during diverse circumstances and seasons is necessary. Last but not least, a collaborative investigation involving Marketing and the Call Center is vital to comprehend the dynamics behind this data and refine strategies accordingly.

<u>Applications</u>

The solutions developed in this project, including a predictive machine learning model and a comprehensive feature-wise analysis, can be effectively applied to similar business contexts. Specifically, these solutions are tailored for binary classification problems utilizing customer or call center data.

For optimal applicability, it is recommended to use these solutions for business problems relevant to the marketing and banking sectors. The data utilized in this project was sourced from a call center, but the suitability of the source is not overly restrictive as long as the dataset contains similar types of information.

### <b>Notebook and Installation</b>

For more details, you may refer to <a href='https://github.com/henryhyunwookim/Term-Deposit-Marketing/blob/main/Term%20Deposit%20Marketing.ipynb'>this notebook</a> directly.

To run Term Deposit Marketing.ipynb locally, please clone or fork this repo and install the required packages by running the following command:

pip install -r requirements.txt

##### <i>* Associated with Apziva</i>
