import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

def chunkList (lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def getSongsStats (streamingHistoryToSpotifyTrackInfoJsonFilePath, spotifyTrackInfoToTrackStatsJsonFilePath):
    tracksInfo = [] 
    with open(streamingHistoryToSpotifyTrackInfoJsonFilePath, 'r') as file:
        tracksInfo = json.loads(file.read())

    songsUriNoChunk = [] 
    for trackInfo in tracksInfo:
        songsUriNoChunk.append(trackInfo["trackUri"])

    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id = 'ca5ea105ab4e4630a5acde4d234dd68f',
        client_secret = 'fb5965d73cc64016b68b08b018f92911'
    ))

    stats = []
    requestNumber = 0
    songsUri100Chunk = chunkList (songsUriNoChunk, 100) #100 is the max number of elements per query 
    
    for songsUri in songsUri100Chunk:
        requestNumber += 1
        if requestNumber%10 == 0:
            time.sleep(1.5)
        stats.extend (spotify.audio_features (songsUri))

    with open(spotifyTrackInfoToTrackStatsJsonFilePath, "w") as write_file:
        json.dump(stats, write_file)
