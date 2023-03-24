import json
import time

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def chunkList(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def getSongsStats(streamingHistoryToSpotifyTrackInfoJsonFilePath, spotifyTrackInfoToTrackStatsJsonFilePath):
    tracksInfo = []
    with open(streamingHistoryToSpotifyTrackInfoJsonFilePath, 'r') as file:
        tracksInfo = json.loads(file.read())

    songsUriNoChunk = []
    for trackInfo in tracksInfo:
        songsUriNoChunk.append(trackInfo["trackUri"])

    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id='client_id',
        client_secret='client_secret'
    ))

    stats = []
    requestNumber = 0
    # 100 is the max number of elements per query
    songsUri100Chunk = chunkList(songsUriNoChunk, 100)

    for songsUri in songsUri100Chunk:
        requestNumber += 1
        if requestNumber % 10 == 0:
            time.sleep(1.5)
        stats.extend(spotify.audio_features(songsUri))

    with open(spotifyTrackInfoToTrackStatsJsonFilePath, "w") as write_file:
        json.dump(stats, write_file)
