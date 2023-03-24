from getSongsReleasedYesterday import getSongsReleasedYesterday
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import json 
import csv

def chunkList (lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]


yesterdaySongs = getSongsReleasedYesterday ()

yesterDaySongsDatas = []

for song in yesterdaySongs:
    yesterDaySongsDatas.append({
        "trackUri": song["uri"],
        "trackIsexplicit": song["explicit"],
    })

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id = 'client_id',
        client_secret = 'client_secret'
    ))


songsUriNoChunk = []
for song in yesterDaySongsDatas:
    songsUriNoChunk.append(song["trackUri"])


songsUri100Chunk = chunkList (songsUriNoChunk, 100) #100 is the max number of elements per query 
requestNumber = 0
yesterDaySongsStats = []
for songsUri in songsUri100Chunk:
    requestNumber += 1
    if requestNumber%10 == 0:
        time.sleep(1.5)
    yesterDaySongsStats.extend (spotify.audio_features (songsUri))

if (len (yesterDaySongsDatas) == len (yesterDaySongsStats)):
    outputData = []
    for singleStreamingHistoryToSpotifyTrackInfo, singleSpotifyTrackInfoToTrackStats in zip(yesterDaySongsDatas, yesterDaySongsStats):
        #sometimes spotipy.Spotifyaudio_features (songUri) can't find song stats so there are null values in SpotifyTrackInfoToTrackStats
        #example:  
        #songUri = "spotify:track:7LnDKGTwMoLv3XSgxOnj83"
        #spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        #        client_id = 'client_id',
        #        client_secret = 'client_secret'
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
                }
            )

    with open("spotifyDataExtraction/testDataSetCreation/dataSetTest.json", 'w') as f:
        json.dump(outputData, f)

    with open("spotifyDataExtraction/testDataSetCreation/dataSetTest.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(outputData[0].keys())

            for row in outputData:
                int_row = [int(float(cell)) for cell in row.values()]
                writer.writerow(int_row)

else:
    print ("error in the making of the datas, the 2 json files have different sizes")
