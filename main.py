#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 23:07:28 2025
@author: arpita
"""
# main.py
import sys
import os
# Import from existing files
from advanced_emotion_detection1 import AdvancedEmotionDetector
from sentimental_sae import recommend_movies_two_categories

def display_recommendations(popular_recs, hidden_gems):
    """Display movie recommendations in two categories"""
    print("\n" + "=" * 50)
    print("POPULAR RECOMMENDATIONS")
    print("=" * 50)
    if popular_recs:
        for i, (movie_id, title, genres, rating, count) in enumerate(popular_recs):
            print(f"{i+1}. {title}")
            print(f"   Genres: {genres}")
            print(f"   Rating: {rating:.1f}/5 ({count} ratings)")
    else:
        print("No popular recommendations found for your current mood.")
    
    print("\n" + "=" * 50)
    print("HIDDEN GEMS ")
    print("=" * 50)
    if hidden_gems:
        for i, (movie_id, title, genres, rating, count) in enumerate(hidden_gems):
            print(f"{i+1}. {title}")
            print(f"   Genres: {genres}")
            print(f"   Rating: {rating:.1f}/5 ({count} ratings)")
    else:
        print("No hidden gems found for your current mood.")

def main():
    is_first = True
    while True:
        x = "Movie night is coming up!"+ "\n" + "How's your day rolling along?"
        y = 'Type "Exit" to finish recommendations' + "\n" + "I'm listening! Let's unwind your day together."
        print('\n\n')
        if is_first: 
            print(x)
            is_first = False
        else: print(y)
        
        
        user_input = input()
        
        if user_input.lower() == 'exit':
            print("\nThank you for using Mood Movie Recommender. Enjoy your movie night!")
            break
        
        try:
            # Detect emotion from user input
            print("\nAnalyzing your mood...")
            detector = AdvancedEmotionDetector()
            emotion_result = detector.detect(user_input)
            
            detected_emotion = emotion_result['emotion']
            confidence = emotion_result['confidence']
            
            print(f"\nI sense that you're feeling: {detected_emotion} (confidence: {confidence:.4f})")
            print("Top emotions detected:")
            for emotion, score in emotion_result['emotion_ranking'][:3]:
                print(f"  {emotion}: {score:.4f}")
            
            # Get movie recommendations based on emotion with two categories
            print("\nBased on your mood, here are some movie recommendations:")
            popular_recs, hidden_gems = recommend_movies_two_categories(detected_emotion)
            
            if popular_recs or hidden_gems:
                display_recommendations(popular_recs, hidden_gems)
            else:
                print("No specific recommendations found for your current mood.")
                
                # Fall back to a common emotion if no matches
                fallback_emotion = 'joy'
                print(f"\nInstead, here are some general recommendations based on {fallback_emotion}:")
                popular_recs, hidden_gems = recommend_movies_two_categories(fallback_emotion)
                display_recommendations(popular_recs, hidden_gems)
                
        except Exception as e:
            print(f"Error: {e}")
            print("Could not process your request. Please try again.")

if __name__ == "__main__":
    main()