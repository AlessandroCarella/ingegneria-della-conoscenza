import json
import csv
import math


def normalizeCos1To100 (value):
    return (math.cos(value) + 1) / 2 * 100

def getValue (singleStreamingHistoryToSpotifyTrackInfo):
    value = (
        singleStreamingHistoryToSpotifyTrackInfo["msPlayed"]
                                /    
        singleStreamingHistoryToSpotifyTrackInfo["trackDuration"]
        )
    return normalizeCos1To100(value)

def findMedianValueForKey (singleStreamingHistoryToSpotifyTrackInfo, key):
    values = [obj[key] for obj in singleStreamingHistoryToSpotifyTrackInfo]
    return sum (values)/len (values)

def getJsonList (fileName):
    with open(fileName, 'r') as file:
        jsonList = json.loads(file.read())
    return jsonList

def jsonsWithNeededFeaturesToCvs ():
    StreamingHistoryToSpotifyTrackInfo = getJsonList ('spotifyDataExtraction/trainDataSetCreation/StreamingHistoryToSpotifyTrackInfo.json')

    SpotifyTrackInfoToTrackStats = getJsonList('spotifyDataExtraction/trainDataSetCreation/SpotifyTrackInfoToTrackStats.json')

    if (len (StreamingHistoryToSpotifyTrackInfo) == len (SpotifyTrackInfoToTrackStats)):
        outputData = []
        for singleStreamingHistoryToSpotifyTrackInfo, singleSpotifyTrackInfoToTrackStats in zip(StreamingHistoryToSpotifyTrackInfo, SpotifyTrackInfoToTrackStats):
            #sometimes spotipy.Spotifyaudio_features (songUri) can't find song stats so there are null values in SpotifyTrackInfoToTrackStats
            #example:  
            #songUri = "spotify:track:7LnDKGTwMoLv3XSgxOnj83"
            #spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
            #        client_id = 'ca5ea105ab4e4630a5acde4d234dd68f',
            #        client_secret = 'fb5965d73cc64016b68b08b018f92911'
            #    ))
            #print (spotify.audio_features (songUri)) 
            if singleSpotifyTrackInfoToTrackStats: 
                outputData.append({
                    #from StreamingHistoryToSpotifyTrackInfo:
                    ##"trackUri": normalizeCos1To100(singleStreamingHistoryToSpotifyTrackInfo["trackUri"]), #as id
                    "trackIsexplicit": 1 if singleStreamingHistoryToSpotifyTrackInfo["trackIsexplicit"] else 0, #to check wheter the user prefers explicit tracks
                    #from SpotifyTrackInfoToTrackStats:
                    "danceability": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["danceability"]),
                    "energy": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["energy"]),
                    "key": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["key"]),
                    "loudness": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["loudness"]),
                    "speechiness": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["speechiness"]),
                    "acousticness": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["acousticness"]),
                    "instrumentalness": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["instrumentalness"]),
                    "liveness": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["liveness"]),
                    "valence": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["valence"]),
                    "tempo": normalizeCos1To100(singleSpotifyTrackInfoToTrackStats["tempo"]),
                    #calculated from data
                    "songIsLiked": 1 if getValue (singleStreamingHistoryToSpotifyTrackInfo) >= 50 else 0,
                    }
                )
        
        with open('spotifyDataExtraction/trainDataSetCreation/dataSet.json', 'w') as f:
            json.dump(outputData, f)
        
        with open('spotifyDataExtraction/trainDataSetCreation/dataSet.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(outputData[0].keys())

            for row in outputData:
                int_row = [int(float(cell)) for cell in row.values()]
                writer.writerow(int_row)

    else:
        print ("error in the making of the datas, the 2 json files have different sizes")

jsonsWithNeededFeaturesToCvs ()