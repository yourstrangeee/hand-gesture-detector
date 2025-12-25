import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Tuple, List
import time
from PIL import Image

load_dotenv()


class HybridGestureRecognizer:
    
    def __init__(self):
        """Initialize hybrid gesture recognizer with MediaPipe, local recognition, and optional Gemini AI."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=1
        )
        
        self.gesture_history = deque(maxlen=7)
        self.confidence_history = deque(maxlen=7)
        
        self._setup_gemini()
        
        self.use_ai = False
        self.ai_available = False
        self.last_ai_attempt = 0
        self.ai_cooldown = 60
        self.api_retry_delay = 0
        
    def _setup_gemini(self) -> None:
        """Configure Gemini AI API with key from environment variables."""
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_api_key_here':
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                self.ai_available = True
                print("Gemini AI initialized (optional enhancement)")
            except Exception as e:
                self.model = None
                self.ai_available = False
                print(f"Gemini AI unavailable: {e}")
        else:
            self.model = None
            self.ai_available = False
            print("Gemini AI not configured (using local recognition)")
    
    def _calculate_finger_state(self, landmarks) -> List[int]:
        """
        Calculate which fingers are extended based on landmark positions.
        Returns list of 5 binary values [thumb, index, middle, ring, pinky].
        """
        fingers = []
        
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        
        if thumb_tip.x < landmarks[2].x - 0.05:
            fingers.append(1)
        else:
            fingers.append(0)
        
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        finger_mcps = [5, 9, 13, 17]
        
        for tip_idx, pip_idx, mcp_idx in zip(finger_tips, finger_pips, finger_mcps):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            
            if tip.y < pip.y - 0.02 and tip.y < mcp.y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def _recognize_gesture_local(self, fingers: List[int], landmarks) -> Tuple[str, float]:
        """
        Recognize hand gesture using local algorithm based on finger states and landmarks.
        Returns tuple of (gesture_name, confidence_score).
        """
        thumb, index, middle, ring, pinky = fingers
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        if fingers == [1, 0, 0, 0, 0]:
            if thumb_tip.y < wrist.y - 0.1:
                return 'Thumbs Up', 0.94
            else:
                return 'Thumbs Sideways', 0.88
        
        if fingers == [1, 1, 1, 1, 1]:
            tip_spread = max(
                abs(index_tip.x - pinky_tip.x),
                abs(index_tip.y - pinky_tip.y)
            )
            if tip_spread > 0.15:
                return 'Open Palm', 0.93
            return 'Hand Spread', 0.87
        
        if fingers == [0, 1, 1, 0, 0]:
            index_middle_dist = np.sqrt(
                (index_tip.x - middle_tip.x)**2 + 
                (index_tip.y - middle_tip.y)**2
            )
            if index_middle_dist > 0.05:
                return 'Peace Sign', 0.92
            return 'Two Fingers Together', 0.85
        
        if fingers == [0, 0, 0, 0, 0]:
            return 'Fist', 0.91
        
        if fingers == [0, 1, 0, 0, 0]:
            return 'Pointing', 0.90
        
        if fingers == [1, 0, 0, 1, 1]:
            thumb_index_dist = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            )
            if thumb_index_dist < 0.08:
                return 'OK Sign', 0.91
        
        if fingers == [0, 1, 0, 0, 1]:
            return 'Rock On', 0.89
        
        if fingers == [1, 1, 0, 0, 0]:
            return 'Gun Sign', 0.88
        
        if fingers == [0, 0, 1, 0, 0]:
            return 'Middle Finger', 0.87
        
        if sum(fingers) == 4:
            if thumb == 1:
                return 'Four Fingers (with thumb)', 0.86
            else:
                return 'Four Fingers', 0.86
        
        if sum(fingers) == 3:
            if thumb == 1:
                return 'Three Fingers (with thumb)', 0.85
            else:
                return 'Three Fingers', 0.85
        
        if sum(fingers) == 2:
            return 'Two Fingers', 0.84
        
        if sum(fingers) == 1:
            return 'One Finger', 0.83
        
        return 'Custom Gesture', 0.75
    
    def _analyze_gesture_with_ai(self, hand_region: np.ndarray) -> Tuple[str, float]:
        """
        Use Gemini Vision AI to analyze and identify the hand gesture with rate limit handling.
        Returns tuple of (gesture_name, confidence_score).
        """
        if not self.model or not self.ai_available:
            return None, None
        
        current_time = time.time()
        
        if current_time - self.last_ai_attempt < self.api_retry_delay:
            return None, None
        
        try:
            img = Image.fromarray(cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB))
            
            prompt = """Analyze this hand gesture and respond ONLY in this format:
GESTURE: [specific name]
CONFIDENCE: [0.0-1.0]

Identify any hand gesture including numbers, signs, or custom positions."""

            response = self.model.generate_content([prompt, img])
            response_text = response.text.strip()
            
            self.last_ai_attempt = current_time
            self.api_retry_delay = 0
            
            gesture_name = "Unknown"
            confidence = 0.75
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('GESTURE:'):
                    gesture_name = line.replace('GESTURE:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    conf_str = line.replace('CONFIDENCE:', '').strip()
                    try:
                        confidence = float(conf_str)
                        confidence = max(0.0, min(1.0, confidence))
                    except:
                        confidence = 0.80
            
            return gesture_name, confidence
            
        except Exception as e:
            error_str = str(e)
            
            if '429' in error_str or 'quota' in error_str.lower():
                if 'retry in' in error_str.lower():
                    try:
                        import re
                        match = re.search(r'retry in (\d+)', error_str.lower())
                        if match:
                            self.api_retry_delay = int(match.group(1)) + 5
                    except:
                        self.api_retry_delay = 60
                else:
                    self.api_retry_delay = 60
                
                print(f"API rate limit reached. Cooldown: {self.api_retry_delay}s")
                self.ai_available = False
                time.sleep(1)
            
            self.last_ai_attempt = current_time
            return None, None
    
    def _extract_hand_region(self, frame: np.ndarray, hand_landmarks) -> Optional[np.ndarray]:
        """Extract and crop hand region from frame for AI analysis."""
        h, w, _ = frame.shape
        
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min = max(0, int(min(x_coords) - 50))
        x_max = min(w, int(max(x_coords) + 50))
        y_min = max(0, int(min(y_coords) - 50))
        y_max = min(h, int(max(y_coords) + 50))
        
        if x_max <= x_min or y_max <= y_min:
            return None
        
        hand_region = frame[y_min:y_max, x_min:x_max]
        
        if hand_region.size == 0:
            return None
        
        return hand_region
    
    def _smooth_gesture(self, current_gesture: str, current_confidence: float) -> Tuple[str, float]:
        """
        Apply temporal smoothing to reduce gesture flickering.
        Returns smoothed gesture name and confidence.
        """
        self.gesture_history.append(current_gesture)
        self.confidence_history.append(current_confidence)
        
        if len(self.gesture_history) < 4:
            return current_gesture, current_confidence
        
        gesture_counts = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        most_common_gesture = max(gesture_counts, key=gesture_counts.get)
        
        if gesture_counts[most_common_gesture] >= 4:
            avg_confidence = np.mean([
                conf for g, conf in zip(self.gesture_history, self.confidence_history)
                if g == most_common_gesture
            ])
            return most_common_gesture, avg_confidence
        
        return current_gesture, current_confidence
    
    def _draw_ui(self, frame: np.ndarray, gesture: str, 
                confidence: float, fps: float, hand_detected: bool,
                mode: str) -> np.ndarray:
        """Draw professional UI overlay with gesture info, confidence bar and metrics."""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        
        if hand_detected:
            box_color = (0, 80, 0)
        else:
            box_color = (0, 0, 80)
        
        cv2.rectangle(overlay, (10, 10), (500, 200), box_color, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        status_color = (0, 255, 0) if hand_detected else (100, 100, 255)
        status_text = "Hand Detected" if hand_detected else "No Hand Detected"
        cv2.putText(frame, status_text, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        mode_color = (0, 255, 255) if mode == "AI" else (255, 150, 0)
        cv2.putText(frame, f"Mode: {mode}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        if hand_detected and gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
            
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            
            bar_width = int(460 * confidence)
            cv2.rectangle(frame, (20, 170), (20 + bar_width, 190), 
                        (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 170), (480, 190), (255, 255, 255), 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, "Q: Quit | A: Toggle AI | C: Explain", 
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (200, 200, 200), 1)
        
        return frame
    
    def _get_explanation(self, gesture: str) -> Optional[str]:
        """Get detailed explanation about the gesture from Gemini AI."""
        if not self.model or not self.ai_available:
            return None
        
        try:
            prompt = f"""Briefly explain the '{gesture}' hand gesture in 1-2 sentences: its meaning and common uses."""
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Explanation Error: {e}")
            return None
    
    def run(self) -> None:
        """Main application loop for hybrid gesture recognition."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam")
            return
        
        print("\n" + "="*70)
        print("🚀 Hybrid Hand Gesture Recognition System Active")
        print("="*70)
        print("Mode: Local Recognition (Fast & Reliable)")
        if self.ai_available:
            print("AI Enhancement: Available (Toggle with 'A' key)")
        else:
            print("AI Enhancement: Not configured (Local mode only)")
        print("="*70)
        print("Controls:")
        print("  Q - Quit")
        print("  A - Toggle AI mode (if available)")
        print("  C - Get gesture explanation (AI required)")
        print("="*70 + "\n")
        
        frame_times = deque(maxlen=30)
        frame_count = 0
        ai_check_interval = 30
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            hand_detected = False
            current_gesture = None
            current_confidence = 0.0
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                fingers = self._calculate_finger_state(hand_landmarks.landmark)
                current_gesture, current_confidence = self._recognize_gesture_local(
                    fingers, hand_landmarks.landmark
                )
                
                if self.use_ai and self.ai_available and frame_count % ai_check_interval == 0:
                    hand_region = self._extract_hand_region(frame, hand_landmarks)
                    if hand_region is not None:
                        ai_gesture, ai_conf = self._analyze_gesture_with_ai(hand_region)
                        if ai_gesture and ai_conf:
                            current_gesture = ai_gesture
                            current_confidence = ai_conf
                            print(f"AI: {ai_gesture} ({ai_conf*100:.1f}%)")
                
                current_gesture, current_confidence = self._smooth_gesture(
                    current_gesture, current_confidence
                )
                
                frame_count += 1
            else:
                self.gesture_history.clear()
                self.confidence_history.clear()
                frame_count = 0
            
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
            
            mode = "AI Enhanced" if self.use_ai and self.ai_available else "Local"
            frame = self._draw_ui(frame, current_gesture, current_confidence, 
                                fps, hand_detected, mode)
            
            cv2.imshow('Hybrid Gesture Recognition System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nShutting down gracefully...")
                break
            
            elif key == ord('a') or key == ord('A'):
                if self.ai_available:
                    self.use_ai = not self.use_ai
                    mode_str = "AI Enhanced" if self.use_ai else "Local"
                    print(f"Switched to {mode_str} mode")
                else:
                    print("AI not available")
            
            elif key == ord('c') or key == ord('C'):
                if current_gesture and self.ai_available:
                    print(f"\nGetting explanation for: {current_gesture}")
                    explanation = self._get_explanation(current_gesture)
                    if explanation:
                        print(f"{explanation}\n")
                    else:
                        print("Could not get explanation\n")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Application closed successfully")


def main():
    """Application entry point with environment setup verification."""
    print("\n" + "="*70)
    print("Initializing Hybrid Gesture Recognition System")
    print("="*70)
    
    if not os.path.exists('.env'):
        print("\nCreating .env template...")
        with open('.env', 'w') as f:
            f.write("# Gemini API Configuration (Optional)\n")
            f.write("# Get your API key from: https://makersuite.google.com/app/apikey\n\n")
            f.write("GEMINI_API_KEY=your_api_key_here\n")
        print(".env file created")
        print("Add Gemini API key for AI enhancement (optional)\n")
    
    try:
        recognizer = HybridGestureRecognizer()
        recognizer.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*70)
        print("Thank you for using Hybrid Gesture Recognition System!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()