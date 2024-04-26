from face_embedding.face_embedding_manager import FaceEmbeddingManager
import uuid
import mediapipe as mp

ImageEmbedder = mp.tasks.vision.ImageEmbedder

class Person:
    person_id = None
    first_embedding = None
    embeddings = []
    images = []

    def __init__(self, first_embedding):
        Person.last_person_index = Person.last_person_index if hasattr(Person,"last_person_index") else 1
        self.person_id = Person.last_person_index
        Person.last_person_index += 1
        self.first_embedding = first_embedding
        self.embeddings.append(first_embedding)

    def add_matched_embedding(self, image, embedding):
        self.embeddings.append(embedding)
        self.images.append(image)


class PeopleManager:
    SIMILARITY_THRESHOLD = 0.4

    def __init__(self) -> None:
        self.people_db: list[Person] = []
        self.face_embedding_manager = FaceEmbeddingManager()

    def get_face_id(self, image):
        embedding = self.face_embedding_manager.embed_sync(image).embeddings[0]

        matches = []

        for person in self.people_db:
            similarity = ImageEmbedder.cosine_similarity(
                embedding,
                person.first_embedding
            )
            print(f"{similarity=}")
            if similarity > self.SIMILARITY_THRESHOLD:
                matches.append((similarity, person))

        if len(matches) == 0:
            # handle adding new person
            new_person = Person(first_embedding=embedding)
            self.people_db.append(new_person)
            return new_person.person_id
        
        matches.sort(reverse=True, key=lambda x: x[0])

        matches[0][1].add_matched_embedding(image, embedding)
        return matches[0][1].person_id
        