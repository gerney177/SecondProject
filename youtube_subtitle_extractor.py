import os
import re
import json
from typing import List, Dict, Optional, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import requests
from urllib.parse import urlparse, parse_qs


class YouTubeSubtitleExtractor:
    """
    YouTube subtitle extractor with multilingual support.
    Extracts subtitles from YouTube videos and supports Korean/English languages.
    """
    
    def __init__(self):
        self.formatter = TextFormatter()
        self.supported_languages = ['ko', 'en', 'ja', 'zh', 'es', 'fr', 'de']
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            Optional[str]: Video ID if valid, None otherwise
        """
        try:
            # Handle various YouTube URL formats
            if 'youtu.be/' in url:
                return url.split('youtu.be/')[-1].split('?')[0]
            elif 'youtube.com/watch' in url:
                parsed_url = urlparse(url)
                return parse_qs(parsed_url.query)['v'][0]
            elif 'youtube.com/embed/' in url:
                return url.split('youtube.com/embed/')[-1].split('?')[0]
            else:
                # Assume it's already a video ID
                if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
                    return url
                return None
        except Exception as e:
            print(f"Error extracting video ID: {e}")
            return None
    
    def get_available_languages(self, video_id: str) -> List[str]:
        """
        Get list of available subtitle languages for a video.
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            List[str]: List of available language codes
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            languages = []
            
            for transcript in transcript_list:
                languages.append(transcript.language_code)
                
            return languages
        except Exception as e:
            print(f"Error getting available languages: {e}")
            return []
    
    def extract_subtitles(self, 
                         video_url: str, 
                         language_preference: List[str] = ['ko', 'en']) -> Optional[Dict]:
        """
        Extract subtitles from YouTube video with language preference.
        
        Args:
            video_url (str): YouTube video URL or video ID
            language_preference (List[str]): Preferred languages in order
            
        Returns:
            Optional[Dict]: Dictionary containing subtitles and metadata
        """
        video_id = self.extract_video_id(video_url)
        if not video_id:
            print("Invalid YouTube URL or video ID")
            return None
        
        try:
            # Get available languages
            available_languages = self.get_available_languages(video_id)
            print(f"Available languages: {available_languages}")
            
            # Find the best available language
            selected_language = None
            for lang in language_preference:
                if lang in available_languages:
                    selected_language = lang
                    break
            
            # If no preferred language found, use the first available
            if not selected_language and available_languages:
                selected_language = available_languages[0]
            
            if not selected_language:
                print("No subtitles available for this video")
                return None
            
            print(f"Using language: {selected_language}")
            
            # Extract transcript
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=[selected_language]
            )
            
            # Format transcript text
            formatted_text = ""
            timestamps = []
            
            for entry in transcript:
                formatted_text += entry['text'] + " "
                timestamps.append({
                    'start': entry['start'],
                    'duration': entry['duration'],
                    'text': entry['text']
                })
            
            # Clean up text
            formatted_text = self._clean_text(formatted_text)
            
            return {
                'video_id': video_id,
                'video_url': f"https://youtube.com/watch?v={video_id}",
                'language': selected_language,
                'subtitle_text': formatted_text,
                'timestamps': timestamps,
                'total_duration': max([entry['start'] + entry['duration'] for entry in transcript]),
                'word_count': len(formatted_text.split())
            }
            
        except Exception as e:
            print(f"Error extracting subtitles: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted subtitle text.
        
        Args:
            text (str): Raw subtitle text
            
        Returns:
            str: Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common subtitle artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [음악], [박수] etc.
        text = re.sub(r'\(.*?\)', '', text)  # Remove (웃음) etc.
        
        # Clean up punctuation
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Space before punctuation
        
        return text.strip()
    
    def extract_subtitles_with_timestamps(self, 
                                        video_url: str, 
                                        language_preference: List[str] = ['ko', 'en'],
                                        segment_duration: int = 30) -> Optional[List[Dict]]:
        """
        Extract subtitles segmented by time intervals.
        
        Args:
            video_url (str): YouTube video URL
            language_preference (List[str]): Preferred languages
            segment_duration (int): Duration of each segment in seconds
            
        Returns:
            Optional[List[Dict]]: List of subtitle segments with timestamps
        """
        subtitle_data = self.extract_subtitles(video_url, language_preference)
        if not subtitle_data:
            return None
        
        timestamps = subtitle_data['timestamps']
        segments = []
        current_segment = {
            'start_time': 0,
            'end_time': segment_duration,
            'text': '',
            'word_count': 0
        }
        
        for entry in timestamps:
            entry_start = entry['start']
            entry_end = entry['start'] + entry['duration']
            
            # If entry fits in current segment
            if entry_start < current_segment['end_time']:
                current_segment['text'] += entry['text'] + ' '
                current_segment['word_count'] = len(current_segment['text'].split())
            else:
                # Save current segment and start new one
                if current_segment['text'].strip():
                    current_segment['text'] = self._clean_text(current_segment['text'])
                    segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    'start_time': current_segment['end_time'],
                    'end_time': current_segment['end_time'] + segment_duration,
                    'text': entry['text'] + ' ',
                    'word_count': 0
                }
        
        # Add final segment
        if current_segment['text'].strip():
            current_segment['text'] = self._clean_text(current_segment['text'])
            current_segment['word_count'] = len(current_segment['text'].split())
            segments.append(current_segment)
        
        return segments
    
    def save_subtitles(self, subtitle_data: Dict, output_path: str) -> bool:
        """
        Save subtitle data to JSON file.
        
        Args:
            subtitle_data (Dict): Subtitle data from extract_subtitles
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(subtitle_data, f, ensure_ascii=False, indent=2)
            print(f"Subtitles saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving subtitles: {e}")
            return False


def main():
    """Example usage of YouTubeSubtitleExtractor"""
    extractor = YouTubeSubtitleExtractor()
    
    # Example YouTube URL (replace with actual URL)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("=== YouTube Subtitle Extractor Test ===")
    print(f"Extracting subtitles from: {test_url}")
    
    # Extract subtitles
    result = extractor.extract_subtitles(test_url, language_preference=['ko', 'en'])
    
    if result:
        print(f"\nVideo ID: {result['video_id']}")
        print(f"Language: {result['language']}")
        print(f"Duration: {result['total_duration']:.2f} seconds")
        print(f"Word count: {result['word_count']}")
        print(f"\nSubtitle preview:")
        print(result['subtitle_text'][:200] + "..." if len(result['subtitle_text']) > 200 else result['subtitle_text'])
        
        # Save to file
        extractor.save_subtitles(result, 'subtitle_output.json')
        
        # Test segmented extraction
        segments = extractor.extract_subtitles_with_timestamps(test_url, segment_duration=60)
        if segments:
            print(f"\nExtracted {len(segments)} segments")
            for i, segment in enumerate(segments[:3]):  # Show first 3 segments
                print(f"Segment {i+1}: {segment['start_time']}-{segment['end_time']}s")
                print(f"Text: {segment['text'][:100]}...")
    else:
        print("Failed to extract subtitles")


if __name__ == "__main__":
    main()
