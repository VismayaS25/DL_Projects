from genderize import Genderize
import speech_recognition as sr

def recognize_gender_from_name(name):
    genderize = Genderize()
    gender = genderize.get([name])[0]['gender']
    return gender

def recognize_gender_from_voice():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")

        # Extract name from the recognized text (simple approach)
        name = text.split()[0]

        # Recognize gender from name
        gender = recognize_gender_from_name(name)
        print(f"Predicted gender: {gender}")

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    recognize_gender_from_voice()