import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import datetime
import time

import json

def getNewAlbums (results, releaseDate, spotify, requestNumber):
    outputAlbums = []

    newAlbums = results ["albums"]["items"]
    for album in newAlbums:
        if album: #check for none values
            if album["release_date"] == releaseDate:
                outputAlbums.append(album)

    resultsAlbum = results
    while resultsAlbum['albums']['next']:
        requestNumber += 1
        if requestNumber%10 == 0:
            time.sleep(1.5)
        resultsAlbum = spotify.next(results['albums'])
        newAlbums = resultsAlbum ["albums"]["items"]
        for album in newAlbums:
            if album: #check for none values
                if album["release_date"] == releaseDate:
                    outputAlbums.append(album)
    
    return outputAlbums, requestNumber

def getNewTracks (results, releaseDate, spotify, requestNumber):
    outputTracks = []
    newTracks = results ["tracks"]["items"]
    for track in newTracks:
        if track: #check for none values
            if track["release_date"] == releaseDate:
                outputTracks.append(track)

    resultsTracks = results
    while resultsTracks['tracks']['next']:
        requestNumber += 1
        if requestNumber%10 == 0:
            time.sleep(1.5)
        resultsTracks = spotify.next(results['tracks'])
        newTracks = resultsTracks ["albums"]["items"]
        for track in newTracks:
            if track: #check for none values
                if track["release_date"] == releaseDate:
                    outputTracks.append(track)
    
    return outputTracks, requestNumber

def getTracksFromAlbum (album, spotify, requestNumber):
    # Fetch the album details using the album() method
    requestNumber += 1
    if requestNumber%10 == 0:
        time.sleep(1.5)
    album = spotify.album(album ["uri"])

    # Print the album object
    return album["tracks"]["items"], requestNumber

def getSongsReleasedYesterday ():
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id = 'ca5ea105ab4e4630a5acde4d234dd68f',#ca5ea105ab4e4630a5acde4d234dd68f
        client_secret = 'fb5965d73cc64016b68b08b018f92911'#fb5965d73cc64016b68b08b018f92911
        )
    )

    releaseDate = str(datetime.date.today() - datetime.timedelta(days=1))

    #tag:new filter query by last 2 weeks releases 
    requestNumber = 1
    results = spotify.search(q='tag:new', type='track,album', limit=50)

    outputAlbums, requestNumber = getNewAlbums (results, releaseDate, spotify, requestNumber)
    outputTracks, requestNumber = getNewTracks (results, releaseDate, spotify, requestNumber)
    
    for album in outputAlbums:
        albumTracks, requestNumber = getTracksFromAlbum (album, spotify, requestNumber)
        outputTracks.extend (albumTracks)

    with open("spotifyDataExtraction/testDataSetCreation/yesterdaySongsStats.json", "w") as file:
        json.dump(outputTracks, file)
    
    return outputTracks