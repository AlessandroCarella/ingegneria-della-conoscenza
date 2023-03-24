from stampe import *

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def isSpotifyTrackUrl(url):
    return "open.spotify.com/track" in url

def getAudioFeatures(url):
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id = 'ca5ea105ab4e4630a5acde4d234dd68f',#ca5ea105ab4e4630a5acde4d234dd68f
        client_secret = 'fb5965d73cc64016b68b08b018f92911'#fb5965d73cc64016b68b08b018f92911
    ))

    return spotify.audio_features(url.split('/')[-1].split('?')[0]), spotify.track(url)["explicit"]

def trackAudioFeaturesToQueryEvidence (trackAudioFeatures, explicit):
    return {
        "trackIsexplicit": explicit,
        "danceability": trackAudioFeatures [0]["danceability"],
        "energy": trackAudioFeatures [0]["energy"],
        "key": trackAudioFeatures [0]["key"],
        "loudness": trackAudioFeatures [0]["loudness"],
        "speechiness": trackAudioFeatures [0]["speechiness"],
        "acousticness": trackAudioFeatures [0]["acousticness"],
        "instrumentalness": trackAudioFeatures [0]["instrumentalness"],
        "valence": trackAudioFeatures [0]["valence"],
        "tempo": trackAudioFeatures [0]["tempo"],
    }

def queryOutput (data, differentialColumn, evidence):
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

    # Potential querySong
    querySong = data.query( #0
        show_progress=False,
        variables=[differentialColumn],
        evidence=evidence
    )
    prRed("\nProbability for given url:")
    print(querySong)

def querySystem(data, differentialColumn):
    prYellow(
        "\n\n\t\t\t\t\tWelcome to my system!\n\n\t"
        "It allows you to predict whether, taken a song, the user the data are generated on would like it or not.\n\n"
    )

    while True:
        i = 0
        #try:
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
        #except ValueError:
        #print("Wrong input")