import json
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

def getTrackUri (artist_name, track_name):
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id = 'ca5ea105ab4e4630a5acde4d234dd68f',#ca5ea105ab4e4630a5acde4d234dd68f
        client_secret = 'fb5965d73cc64016b68b08b018f92911'#fb5965d73cc64016b68b08b018f92911
    ))

    results = spotify.search(q=f'artist:{artist_name} track:{track_name}', type='track')
    # Get the URI of the first track in the search results
    if results['tracks']['items']:
        if ['tracks']['items']:
            return results['tracks']['items'][0]['uri'], results['tracks']['items'][0]['duration_ms'], results['tracks']['items'][0]['explicit'], results['tracks']['items'][0]['popularity']
    else:
        return "", "", "", ""

def streamingHistoryJsonToSongUriAndMsPlayed ():
    with open(os.path.join (os.path.abspath ("spotifyDataExtraction/trainDataSetCreation"), "StreamingHistory.json"), 'r', encoding="utf-8") as f:
        data = json.load(f)

    for element in data:
        del element['endTime']

    output = []
    
    requestNumber = 0

    for element in data:
        requestNumber += 1
        if requestNumber%10 == 0:
            time.sleep(1.5)

        with open('StreamingHistoryToSpotifyTrackInfo.json', 'w') as f:
            json.dump(output, f)

        trackUri, trackDuration, trackIsexplicit, trackPopularity = getTrackUri (element ["artistName"],  element ["trackName"])
        
        trackAlredyAdded = False
        for track in output:
            if track ['trackUri'] == trackUri:
                track ['timesPlayed'] = track['timesPlayed'] + 1
                track ['msPlayed'] = track["msPlayed"] + element["msPlayed"]
                trackAlredyAdded = True
                break

        if not trackAlredyAdded:
            output.append ({'trackUri': trackUri, 
                'trackDuration': trackDuration, 
                'timesPlayed': 1, 
                'msPlayed':element["msPlayed"], 
                'trackIsexplicit':trackIsexplicit,
                'trackPopularity':trackPopularity
            })

        print (str(element ["artistName"]) + " " + str(element ["trackName"]))
        if trackUri == "":
            print ("NON trovato URI")
    
    with open('StreamingHistoryToSpotifyTrackInfo.json', 'w') as f:
        json.dump(output, f)
    