{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Project 3\n",
    "\n",
    "\n",
    "# Movie Genre Classification\n",
    "\n",
    "Classify a movie genre based on its plot.\n",
    "\n",
    "<img src=\"moviegenre.png\"\n",
    "     style=\"float: left; margin-right: 10px;\" />\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "https://www.kaggle.com/c/miia4201-202019-p3-moviegenreclassification/overview\n",
    "\n",
    "### Data\n",
    "\n",
    "Input:\n",
    "- movie plot\n",
    "\n",
    "Output:\n",
    "Probability of the movie belong to each genre\n",
    "\n",
    "\n",
    "### Evaluation\n",
    "\n",
    "- 20% API\n",
    "- 30% Report with all the details of the solution, the analysis and the conclusions. The report cannot exceed 10 pages, must be send in PDF format and must be self-contained.\n",
    "- 50% Performance in the Kaggle competition (The grade for each group will be proportional to the ranking it occupies in the competition. The group in the first place will obtain 5 points, for each position below, 0.25 points will be subtracted, that is: first place: 5 points, second: 4.75 points, third place: 4.50 points ... eleventh place: 2.50 points, twelfth place: 2.25 points).\n",
    "\n",
    "• The project must be carried out in the groups assigned for module 4.\n",
    "• Use clear and rigorous procedures.\n",
    "• The delivery of the project is on July 12, 2020, 11:59 pm, through Sicua + (Upload: the API and the report in PDF format).\n",
    "• No projects will be received after the delivery time or by any other means than the one established. \n",
    "\n",
    "\n",
    "\n",
    "\n",
    "### Acknowledgements\n",
    "\n",
    "We thank Professor Fabio Gonzalez, Ph.D. and his student John Arevalo for providing this dataset.\n",
    "\n",
    "See https://arxiv.org/abs/1702.01992"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sample Submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.preprocessing import MultiLabelBinarizer\n",
    "from sklearn.multiclass import OneVsRestClassifier\n",
    "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n",
    "from sklearn.metrics import r2_score, roc_auc_score\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataTraining = pd.read_csv('https://github.com/albahnsen/AdvancedMethodsDataAnalysisClass/raw/master/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)\n",
    "dataTesting = pd.read_csv('https://github.com/albahnsen/AdvancedMethodsDataAnalysisClass/raw/master/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>year</th>\n",
       "      <th>title</th>\n",
       "      <th>plot</th>\n",
       "      <th>genres</th>\n",
       "      <th>rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>3107</td>\n",
       "      <td>2003</td>\n",
       "      <td>Most</td>\n",
       "      <td>most is the story of a single father who takes...</td>\n",
       "      <td>['Short', 'Drama']</td>\n",
       "      <td>8.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>900</td>\n",
       "      <td>2008</td>\n",
       "      <td>How to Be a Serial Killer</td>\n",
       "      <td>a serial killer decides to teach the secrets o...</td>\n",
       "      <td>['Comedy', 'Crime', 'Horror']</td>\n",
       "      <td>5.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6724</td>\n",
       "      <td>1941</td>\n",
       "      <td>A Woman's Face</td>\n",
       "      <td>in sweden ,  a female blackmailer with a disfi...</td>\n",
       "      <td>['Drama', 'Film-Noir', 'Thriller']</td>\n",
       "      <td>7.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4704</td>\n",
       "      <td>1954</td>\n",
       "      <td>Executive Suite</td>\n",
       "      <td>in a friday afternoon in new york ,  the presi...</td>\n",
       "      <td>['Drama']</td>\n",
       "      <td>7.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2582</td>\n",
       "      <td>1990</td>\n",
       "      <td>Narrow Margin</td>\n",
       "      <td>in los angeles ,  the editor of a publishing h...</td>\n",
       "      <td>['Action', 'Crime', 'Thriller']</td>\n",
       "      <td>6.6</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      year                      title  \\\n",
       "3107  2003                       Most   \n",
       "900   2008  How to Be a Serial Killer   \n",
       "6724  1941             A Woman's Face   \n",
       "4704  1954            Executive Suite   \n",
       "2582  1990              Narrow Margin   \n",
       "\n",
       "                                                   plot  \\\n",
       "3107  most is the story of a single father who takes...   \n",
       "900   a serial killer decides to teach the secrets o...   \n",
       "6724  in sweden ,  a female blackmailer with a disfi...   \n",
       "4704  in a friday afternoon in new york ,  the presi...   \n",
       "2582  in los angeles ,  the editor of a publishing h...   \n",
       "\n",
       "                                  genres  rating  \n",
       "3107                  ['Short', 'Drama']     8.0  \n",
       "900        ['Comedy', 'Crime', 'Horror']     5.6  \n",
       "6724  ['Drama', 'Film-Noir', 'Thriller']     7.2  \n",
       "4704                           ['Drama']     7.4  \n",
       "2582     ['Action', 'Crime', 'Thriller']     6.6  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataTraining.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>year</th>\n",
       "      <th>title</th>\n",
       "      <th>plot</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1999</td>\n",
       "      <td>Message in a Bottle</td>\n",
       "      <td>who meets by fate ,  shall be sealed by fate ....</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1978</td>\n",
       "      <td>Midnight Express</td>\n",
       "      <td>the true story of billy hayes ,  an american c...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>1996</td>\n",
       "      <td>Primal Fear</td>\n",
       "      <td>martin vail left the chicago da ' s office to ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>1950</td>\n",
       "      <td>Crisis</td>\n",
       "      <td>husband and wife americans dr .  eugene and mr...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>1959</td>\n",
       "      <td>The Tingler</td>\n",
       "      <td>the coroner and scientist dr .  warren chapin ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   year                title  \\\n",
       "1  1999  Message in a Bottle   \n",
       "4  1978     Midnight Express   \n",
       "5  1996          Primal Fear   \n",
       "6  1950               Crisis   \n",
       "7  1959          The Tingler   \n",
       "\n",
       "                                                plot  \n",
       "1  who meets by fate ,  shall be sealed by fate ....  \n",
       "4  the true story of billy hayes ,  an american c...  \n",
       "5  martin vail left the chicago da ' s office to ...  \n",
       "6  husband and wife americans dr .  eugene and mr...  \n",
       "7  the coroner and scientist dr .  warren chapin ...  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataTesting.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create count vectorizer\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7895, 1000)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vect = CountVectorizer(max_features=1000)\n",
    "X_dtm = vect.fit_transform(dataTraining['plot'])\n",
    "X_dtm.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['able', 'about', 'accepts', 'accident', 'accidentally', 'across', 'act', 'action', 'actor', 'actress', 'actually', 'adam', 'adult', 'adventure', 'affair', 'after', 'again', 'against', 'age', 'agent', 'agents', 'ago', 'agrees', 'air', 'alan', 'alex', 'alice', 'alien', 'alive', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'america', 'american', 'among', 'an', 'and', 'angeles', 'ann', 'anna', 'another', 'any', 'anyone', 'anything', 'apartment']\n"
     ]
    }
   ],
   "source": [
    "print(vect.get_feature_names()[:50])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataTraining['genres'] = dataTraining['genres'].map(lambda x: eval(x))\n",
    "\n",
    "le = MultiLabelBinarizer()\n",
    "y_genres = le.fit_transform(dataTraining['genres'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 1, 0, 0],\n",
       "       ...,\n",
       "       [0, 1, 0, ..., 0, 0, 0],\n",
       "       [0, 1, 1, ..., 0, 0, 0],\n",
       "       [0, 1, 1, ..., 0, 0, 0]])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_genres"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train_genres, y_test_genres = train_test_split(X_dtm, y_genres, test_size=0.33, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train multi-class multi-label model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = OneVsRestClassifier(RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=10, random_state=42))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "OneVsRestClassifier(estimator=RandomForestClassifier(bootstrap=True,\n",
       "                                                     class_weight=None,\n",
       "                                                     criterion='gini',\n",
       "                                                     max_depth=10,\n",
       "                                                     max_features='auto',\n",
       "                                                     max_leaf_nodes=None,\n",
       "                                                     min_impurity_decrease=0.0,\n",
       "                                                     min_impurity_split=None,\n",
       "                                                     min_samples_leaf=1,\n",
       "                                                     min_samples_split=2,\n",
       "                                                     min_weight_fraction_leaf=0.0,\n",
       "                                                     n_estimators=100,\n",
       "                                                     n_jobs=-1, oob_score=False,\n",
       "                                                     random_state=42, verbose=0,\n",
       "                                                     warm_start=False),\n",
       "                    n_jobs=None)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.fit(X_train, y_train_genres)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_genres = clf.predict_proba(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7812262183677007"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roc_auc_score(y_test_genres, y_pred_genres, average='macro')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Predict the testing dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_dtm = vect.transform(dataTesting['plot'])\n",
    "\n",
    "cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',\n",
    "        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',\n",
    "        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']\n",
    "\n",
    "y_pred_test_genres = clf.predict_proba(X_test_dtm)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "res = pd.DataFrame(y_pred_test_genres, index=dataTesting.index, columns=cols)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>p_Action</th>\n",
       "      <th>p_Adventure</th>\n",
       "      <th>p_Animation</th>\n",
       "      <th>p_Biography</th>\n",
       "      <th>p_Comedy</th>\n",
       "      <th>p_Crime</th>\n",
       "      <th>p_Documentary</th>\n",
       "      <th>p_Drama</th>\n",
       "      <th>p_Family</th>\n",
       "      <th>p_Fantasy</th>\n",
       "      <th>...</th>\n",
       "      <th>p_Musical</th>\n",
       "      <th>p_Mystery</th>\n",
       "      <th>p_News</th>\n",
       "      <th>p_Romance</th>\n",
       "      <th>p_Sci-Fi</th>\n",
       "      <th>p_Short</th>\n",
       "      <th>p_Sport</th>\n",
       "      <th>p_Thriller</th>\n",
       "      <th>p_War</th>\n",
       "      <th>p_Western</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>0.143030</td>\n",
       "      <td>0.101960</td>\n",
       "      <td>0.024454</td>\n",
       "      <td>0.029938</td>\n",
       "      <td>0.354552</td>\n",
       "      <td>0.138830</td>\n",
       "      <td>0.030787</td>\n",
       "      <td>0.490140</td>\n",
       "      <td>0.073159</td>\n",
       "      <td>0.101339</td>\n",
       "      <td>...</td>\n",
       "      <td>0.025069</td>\n",
       "      <td>0.063208</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.362818</td>\n",
       "      <td>0.056648</td>\n",
       "      <td>0.008970</td>\n",
       "      <td>0.017522</td>\n",
       "      <td>0.202605</td>\n",
       "      <td>0.033989</td>\n",
       "      <td>0.018117</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>0.122624</td>\n",
       "      <td>0.085786</td>\n",
       "      <td>0.024213</td>\n",
       "      <td>0.084795</td>\n",
       "      <td>0.370949</td>\n",
       "      <td>0.216657</td>\n",
       "      <td>0.080359</td>\n",
       "      <td>0.515684</td>\n",
       "      <td>0.062976</td>\n",
       "      <td>0.067019</td>\n",
       "      <td>...</td>\n",
       "      <td>0.024734</td>\n",
       "      <td>0.060935</td>\n",
       "      <td>0.000477</td>\n",
       "      <td>0.149703</td>\n",
       "      <td>0.058190</td>\n",
       "      <td>0.014248</td>\n",
       "      <td>0.020099</td>\n",
       "      <td>0.204794</td>\n",
       "      <td>0.030438</td>\n",
       "      <td>0.018506</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>0.151364</td>\n",
       "      <td>0.110284</td>\n",
       "      <td>0.013762</td>\n",
       "      <td>0.075334</td>\n",
       "      <td>0.304837</td>\n",
       "      <td>0.448736</td>\n",
       "      <td>0.021010</td>\n",
       "      <td>0.611544</td>\n",
       "      <td>0.081741</td>\n",
       "      <td>0.169121</td>\n",
       "      <td>...</td>\n",
       "      <td>0.044538</td>\n",
       "      <td>0.261372</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.335987</td>\n",
       "      <td>0.128505</td>\n",
       "      <td>0.001016</td>\n",
       "      <td>0.048658</td>\n",
       "      <td>0.423242</td>\n",
       "      <td>0.052693</td>\n",
       "      <td>0.025351</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>0.154448</td>\n",
       "      <td>0.125772</td>\n",
       "      <td>0.020991</td>\n",
       "      <td>0.064124</td>\n",
       "      <td>0.340779</td>\n",
       "      <td>0.140892</td>\n",
       "      <td>0.009133</td>\n",
       "      <td>0.632038</td>\n",
       "      <td>0.068287</td>\n",
       "      <td>0.063631</td>\n",
       "      <td>...</td>\n",
       "      <td>0.131074</td>\n",
       "      <td>0.088418</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.197224</td>\n",
       "      <td>0.132208</td>\n",
       "      <td>0.001432</td>\n",
       "      <td>0.039743</td>\n",
       "      <td>0.269385</td>\n",
       "      <td>0.077607</td>\n",
       "      <td>0.017862</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>0.175143</td>\n",
       "      <td>0.210069</td>\n",
       "      <td>0.035476</td>\n",
       "      <td>0.032505</td>\n",
       "      <td>0.313850</td>\n",
       "      <td>0.243150</td>\n",
       "      <td>0.021793</td>\n",
       "      <td>0.427885</td>\n",
       "      <td>0.079781</td>\n",
       "      <td>0.143879</td>\n",
       "      <td>...</td>\n",
       "      <td>0.023859</td>\n",
       "      <td>0.090359</td>\n",
       "      <td>0.000048</td>\n",
       "      <td>0.205117</td>\n",
       "      <td>0.241663</td>\n",
       "      <td>0.002634</td>\n",
       "      <td>0.018403</td>\n",
       "      <td>0.259465</td>\n",
       "      <td>0.021569</td>\n",
       "      <td>0.017585</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   p_Action  p_Adventure  p_Animation  p_Biography  p_Comedy   p_Crime  \\\n",
       "1  0.143030     0.101960     0.024454     0.029938  0.354552  0.138830   \n",
       "4  0.122624     0.085786     0.024213     0.084795  0.370949  0.216657   \n",
       "5  0.151364     0.110284     0.013762     0.075334  0.304837  0.448736   \n",
       "6  0.154448     0.125772     0.020991     0.064124  0.340779  0.140892   \n",
       "7  0.175143     0.210069     0.035476     0.032505  0.313850  0.243150   \n",
       "\n",
       "   p_Documentary   p_Drama  p_Family  p_Fantasy  ...  p_Musical  p_Mystery  \\\n",
       "1       0.030787  0.490140  0.073159   0.101339  ...   0.025069   0.063208   \n",
       "4       0.080359  0.515684  0.062976   0.067019  ...   0.024734   0.060935   \n",
       "5       0.021010  0.611544  0.081741   0.169121  ...   0.044538   0.261372   \n",
       "6       0.009133  0.632038  0.068287   0.063631  ...   0.131074   0.088418   \n",
       "7       0.021793  0.427885  0.079781   0.143879  ...   0.023859   0.090359   \n",
       "\n",
       "     p_News  p_Romance  p_Sci-Fi   p_Short   p_Sport  p_Thriller     p_War  \\\n",
       "1  0.000000   0.362818  0.056648  0.008970  0.017522    0.202605  0.033989   \n",
       "4  0.000477   0.149703  0.058190  0.014248  0.020099    0.204794  0.030438   \n",
       "5  0.000000   0.335987  0.128505  0.001016  0.048658    0.423242  0.052693   \n",
       "6  0.000000   0.197224  0.132208  0.001432  0.039743    0.269385  0.077607   \n",
       "7  0.000048   0.205117  0.241663  0.002634  0.018403    0.259465  0.021569   \n",
       "\n",
       "   p_Western  \n",
       "1   0.018117  \n",
       "4   0.018506  \n",
       "5   0.025351  \n",
       "6   0.017862  \n",
       "7   0.017585  \n",
       "\n",
       "[5 rows x 24 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "res.to_csv('pred_genres_text_RF.csv', index_label='ID')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}