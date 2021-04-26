### Custom definitions and classes if any ###
import pickle
import pandas as pd
import numpy

def predictRuns(testInput):
    prediction = 0
    # loading models
    with open('xgbr.pkl', 'rb') as f:
        xgb_r = pickle.load(f)
    with open('rf.pkl', 'rb') as f:
        rf_r = pickle.load(f)
    # loading encoders
    with open('venue_le.pkl', 'rb') as f:
        venue_le = pickle.load(f)
    with open('team_le.pkl', 'rb') as f:
        team_le = pickle.load(f)
    # loading submission file
    df = pd.read_csv(testInput)
    venue = df.venue.values[0]
    batting_team = df.batting_team.values[0]
    bowling_team = df.bowling_team.values[0]
    innings = df.innings.values[0]
    # getting codes
    if venue in venue_le.classes_:
        venue_code = venue_le.transform([venue])
    else:
        venue_code = list(range(len(venue_le.classes_)))
    if batting_team in team_le.classes_:
        batting_code = team_le.transform([batting_team])
    else:
        batting_code = list(range(len(team_le.classes_)))
    if bowling_team in team_le.classes_:
        bowling_code = team_le.transform([bowling_team])
    else:
        bowling_code = list(range(len(team_le.classes_)))
    # making predictions
    pred_count = 0
    for vc in venue_code:
        for btc in batting_code:
            for bwc in bowling_code:
                prediction += rf_r.predict([[vc, btc, bwc, innings]])[0]
                prediction += xgb_r.predict(numpy.array([[vc, btc, bwc, innings]])).tolist()[0]
                pred_count+=2
    prediction /= pred_count
    return prediction
