from stampe import *

import math

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def isSpotifyTrackUrl(url):
    return "open.spotify.com/track" in url

def getAudioFeatures(url):
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id = 'client_id',#client_id
        client_secret = 'client_secret'#client_secret
    ))

    return spotify.audio_features(url.split('/')[-1].split('?')[0]), spotify.track(url)["explicit"]

def normalizeCos1To100Int (value):
    return int((math.cos(value) + 1) / 2 * 100)

def trackAudioFeaturesToQueryEvidence (trackAudioFeatures, explicit):
    return {
        "trackIsexplicit": 1 if explicit else 0,
        "danceability": normalizeCos1To100Int(trackAudioFeatures [0]["danceability"]),
        "energy": normalizeCos1To100Int(trackAudioFeatures [0]["energy"]),
        "key": normalizeCos1To100Int(trackAudioFeatures [0]["key"]),
        "loudness": normalizeCos1To100Int(trackAudioFeatures [0]["loudness"]),
        "speechiness": normalizeCos1To100Int(trackAudioFeatures [0]["speechiness"]),
        "acousticness": normalizeCos1To100Int(trackAudioFeatures [0]["acousticness"]),
        "instrumentalness": normalizeCos1To100Int(trackAudioFeatures [0]["instrumentalness"]),
        "valence": normalizeCos1To100Int(trackAudioFeatures [0]["valence"]),
        "tempo": normalizeCos1To100Int(trackAudioFeatures [0]["tempo"]),
    }

def queryOutput (data, differentialColumn, evidence):
    querySong = data.query( #0
        show_progress=False,
        variables=[differentialColumn],
        evidence=evidence
    )
    prRed("\nProbability for given url:")
    print(querySong)

def outputExplain ():
    print(
        "+-------------------------------+---------------------------+\n",
        "|         feature name          |    feature probability    |\n",
        "+===============================+===========================+\n",
        "|  feature name(not liked song) | probability feature (val) |\n",
        "+-------------------------------+---------------------------+\n",
        "|   feature name(liked song)    | probability feature (val) |\n",
        "+-------------------------------+---------------------------+",
    )
    prPurple(
        "Probability value fluctuates between 0 (impossible event) to 1 (certain event)\n"
    )

def querySystem(data, differentialColumn):
    prYellow(
        "\n\n\t\t\t\t\tWelcome to my system!\n\n\t"
        "It allows you to predict whether, taken a song, the user the data are generated on would like it or not.\n\n"
    )

    outputExplain ()

    while True:
        i = 0
        prYellow(
            "Input a spotify url for a prediction? - (Typing 'n' close program)"
        )
        result = str(input())
        if "N" == result or result == "n":
            exit(1)
        else:
            if isSpotifyTrackUrl (result):
                trackAudioFeatures, explicit = getAudioFeatures (result)
                if trackAudioFeatures:
                    evidence = trackAudioFeaturesToQueryEvidence (trackAudioFeatures, explicit)
                    queryOutput (data, differentialColumn, evidence)
                else:
                    print ("Couldn't retreive audio features for the given song from spotify api ")
            else:
                print ("Url in input is not a valid spotify track url")