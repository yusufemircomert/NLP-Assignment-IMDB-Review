# Outputs of Naive Bayes:

print(nb.total_pos_words)\
print(nb.total_neg_words)\
print(nb.vocab_size)\
print(nb.prior_pos)\
print(nb.prior_neg)\
print(nb.pos_counter["great"])\
print(nb.neg_counter["great"])

1575152\
1516208\
74002\
0.5\
0.5\
6419\
2642
--------------------------------------

prediction1 = nb.predict(test_df.iloc[0]["text"])\
prediction2 = nb.predict("This movie will be place at 1st in my favourite movies!")\
prediction3 = nb.predict("I couldn't wait for the movie to end, so I, turned it off halfway through. :D It was a complete disappointment.")

#Examples : Example 3\
print(f"{'Positive' if prediction1[0] == 1 else 'Negative'}")\
print(prediction1)

print(f"{'Positive' if prediction2[0] == 1 else 'Negative'}")\
print(prediction2)

print(f"{'Positive' if prediction3[0] == 1 else 'Negative'}")\
print(prediction3)

Negative\
(0, -1167.5758675517511, -1146.4479616999306)\
Positive\
(1, -36.43364380516184, -37.068841883770205)\
Negative\
(0, -57.05497089563332, -53.21115758896025)
-----------------------------------------------
print(f"Accuracy: {accuracy:.5f}")\
Accuracy: 0.82464
--------------------------------
# Outputs Of Logistic Regression

scores = bias_scores(train_df)

print(scores[:2])\
print(scores[-2:])

[(’worst’, 252, 2480, 2732, 6.453036011602796), (’waste’, 99, 1359, 1458, 6.295524245429657)]\
[(’complimented’, 10, 3, 13, 1.3811265770946735), (’conformity’, 10, 3, 13, 1.3811265770946735)]


Result:
![plot](https://github.com/yusufemircomert/NLP-Assignment-IMDB-Review/blob/main/LR_Result.png?raw=true)
