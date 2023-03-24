import json
import csv
import math


def getValue (singleStreamingHistoryToSpotifyTrackInfo, medialValueForTimesPlayed):
    value = (
        (medialValueForTimesPlayed * singleStreamingHistoryToSpotifyTrackInfo["msPlayed"]) 
                                    /    
                singleStreamingHistoryToSpotifyTrackInfo["trackDuration"]
        )
    return (math.cos(value) + 1) / 2 * 100

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

    medialValueForTimesPlayed = findMedianValueForKey (StreamingHistoryToSpotifyTrackInfo, "timesPlayed")

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
                    ##"trackUri": singleStreamingHistoryToSpotifyTrackInfo["trackUri"], #as id
                    "trackIsexplicit": 1 if singleStreamingHistoryToSpotifyTrackInfo["trackIsexplicit"] else 0, #to check wheter the user prefers explicit tracks
                    #from SpotifyTrackInfoToTrackStats: (* 1000 to have only int values (not all values become integers but that makes little difference))
                    "danceability": singleSpotifyTrackInfoToTrackStats["danceability"] * 1000,
                    "energy": singleSpotifyTrackInfoToTrackStats["energy"] * 1000,
                    "key": singleSpotifyTrackInfoToTrackStats["key"] * 1000,
                    "loudness": singleSpotifyTrackInfoToTrackStats["loudness"] * 1000,
                    "speechiness": singleSpotifyTrackInfoToTrackStats["speechiness"] * 1000,
                    "acousticness": singleSpotifyTrackInfoToTrackStats["acousticness"] * 1000,
                    "instrumentalness": singleSpotifyTrackInfoToTrackStats["instrumentalness"] * 1000,
                    "liveness": singleSpotifyTrackInfoToTrackStats["liveness"] * 1000,
                    "valence": singleSpotifyTrackInfoToTrackStats["valence"] * 1000,
                    "tempo": singleSpotifyTrackInfoToTrackStats["tempo"] * 1000,
                    #calculated from data
                    "songIsLiked": 1 if getValue (singleStreamingHistoryToSpotifyTrackInfo, medialValueForTimesPlayed) >= 50 else 0,
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