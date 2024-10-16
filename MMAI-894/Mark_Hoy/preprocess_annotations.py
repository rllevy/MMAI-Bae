import collections
import os
import json
import pandas as pd

# using python 3.10.13

DATA_DIRECTORY = "./data"

CAPTIONS_DIRECTORY = os.path.join(
    DATA_DIRECTORY, "captions_abstract_v002_train-val2015"
)
TRAIN_CAPTIONS = os.path.join(
    CAPTIONS_DIRECTORY, "captions_abstract_v002_train2015.json"
)
VALIDATION_CAPTIONS = os.path.join(
    CAPTIONS_DIRECTORY, "captions_abstract_v002_val2015.json"
)

TRAIN_ANNOTATIONS = os.path.join(
    DATA_DIRECTORY, "abstract_v002_train2015_annotations.json"
)
VALIDATION_ANNOTATION = os.path.join(
    DATA_DIRECTORY, "abstract_v002_val2015_annotations.json"
)

TRAIN_QUESTIONS_DIR = os.path.join(DATA_DIRECTORY, "Questions_Train_abstract_v002")
TRAIN_QUESTIONS_MULTI = os.path.join(
    TRAIN_QUESTIONS_DIR, "MultipleChoice_abstract_v002_train2015_questions.json"
)
TRAIN_QUESTIONS_OPENEND = os.path.join(
    TRAIN_QUESTIONS_DIR, "OpenEnded_abstract_v002_train2015_questions.json"
)
VALIDATION_QUESTIONS_DIR = os.path.join(DATA_DIRECTORY, "Questions_Val_abstract_v002")
VALIDATION_QUESTIONS_MULTI = os.path.join(
    VALIDATION_QUESTIONS_DIR, "MultipleChoice_abstract_v002_val2015_questions.json"
)
VALIDATION_QUESTIONS_OPENEND = os.path.join(
    VALIDATION_QUESTIONS_DIR, "OpenEnded_abstract_v002_val2015_questions.json"
)
TEST_QUESTIONS_DIR = os.path.join(DATA_DIRECTORY, "Questions_Test_abstract_v002")
TEST_QUESTIONS_MULTI = os.path.join(
    TEST_QUESTIONS_DIR, "MultipleChoice_abstract_v002_test2015_questions.json"
)
TEST_QUESTIONS_OPENEND = os.path.join(
    TEST_QUESTIONS_DIR, "OpenEnded_abstract_v002_test2015_questions.json"
)

TRAIN_IMAGE_DIR = os.path.join(DATA_DIRECTORY, "scene_img_abstract_v002_train2015")
VALIDATION_IMAGE_DIR = os.path.join(DATA_DIRECTORY, "scene_img_abstract_v002_val2015")
TEST_IMAGE_DIR = os.path.join(DATA_DIRECTORY, "scene_img_abstract_v002_test2015")

# TRAIN_OUTPUT_FILE = os.path.join(DATA_DIRECTORY, "combined_training.csv")
# VALIDATION_OUTPUT_FILE = os.path.join(DATA_DIRECTORY, "combined_validation.csv")
TRAIN_OUTPUT_FILE = os.path.join(DATA_DIRECTORY, "combined_training_small.csv")
VALIDATION_OUTPUT_FILE = os.path.join(DATA_DIRECTORY, "combined_validation_small.csv")


def extract_captions(caption_json):
    image_filename_by_id = {}
    for image in caption_json["images"]:
        image_filename_by_id[image["image_id"]] = image["file_name"]

    ret_captions = []
    for annotation in caption_json["annotations"]:
        ret_captions.append(
            {
                "id": annotation["id"],
                "image_id": annotation["image_id"],
                "image_filename": image_filename_by_id[annotation["image_id"]],
                "caption": annotation["caption"],
            }
        )
    return ret_captions


def retrieve_captions():
    with open(TRAIN_CAPTIONS, "r", encoding="utf-8") as caption_file:
        training_captions = extract_captions(json.load(caption_file))
    with open(VALIDATION_CAPTIONS, "r", encoding="utf-8") as caption_file:
        validation_captions = extract_captions(json.load(caption_file))
    return training_captions, validation_captions


def combine_annotation_questions(annotation_file, question_files):
    all_questions = {}
    for question_file in question_files:
        with open(question_file, "r", encoding="utf-8") as infile:
            this_question_file = json.load(infile)
            for question in this_question_file["questions"]:
                all_questions[question["question_id"]] = {
                    "image_id": question["image_id"],
                    "question": question["question"],
                }

    ret = []
    with open(annotation_file, "r", encoding="utf-8") as in_annotations:
        annotations = json.load(in_annotations)
        for annotation in annotations["annotations"]:
            question_id = annotation["question_id"]
            if "answers" in annotation and len(annotation["answers"]) > 0:
                # has individual answers
                for answer in annotation["answers"]:
                    ret.append(
                        {
                            "question_id": question_id,
                            "question": all_questions[question_id]["question"],
                            "image_id": annotation["image_id"],
                            "question_type": annotation["question_type"],
                            "answer_type": annotation["answer_type"],
                            "multiple_choice_answer": annotation[
                                "multiple_choice_answer"
                            ],
                            "answer_id": answer["answer_id"],
                            "answer": answer["answer"],
                        }
                    )
            else:
                # pure multiple choice question
                # not sure if this will happen, but just in case
                ret.append(
                    {
                        "question_id": question_id,
                        "question": all_questions[question_id]["question"],
                        "image_id": annotation["image_id"],
                        "question_type": annotation["question_type"],
                        "answer_type": annotation["answer_type"],
                        "multiple_choice_answer": annotation["multiple_choice_answer"],
                        "answer_id": None,
                        "answer": None,
                    }
                )

    return ret


def retrieve_annotations():
    training_annotations = combine_annotation_questions(
        TRAIN_ANNOTATIONS, [TRAIN_QUESTIONS_MULTI, TRAIN_QUESTIONS_OPENEND]
    )

    validation_annotations = combine_annotation_questions(
        VALIDATION_ANNOTATION,
        [VALIDATION_QUESTIONS_MULTI, VALIDATION_QUESTIONS_OPENEND],
    )

    return training_annotations, validation_annotations


def combine_and_output(captions, annotations, output_file):
    columns = [
        "caption_id",
        "image_id",
        "image_filename",
        "caption",
        "question_id",
        "question",
        "question_type",
        "answer_type",
        "multiple_choice_answer",
        "answer_id",
        "answer",
    ]

    captions_by_image_id = collections.defaultdict(list)
    for caption in captions:
        captions_by_image_id[caption["image_id"]].append(caption)

    joined_data = []
    for annotation in annotations:
        annotation_captions = captions_by_image_id[annotation["image_id"]]
        if not annotation_captions:
            # should not happen... but just in case
            continue

        for caption in annotation_captions:
            joined_data.append(
                [
                    caption["id"],
                    caption["image_id"],
                    caption["image_filename"],
                    caption["caption"],
                    annotation["question_id"],
                    annotation["question"],
                    annotation["question_type"],
                    annotation["answer_type"],
                    annotation["multiple_choice_answer"],
                    annotation["answer_id"],
                    annotation["answer"],
                ]
            )

        if len(joined_data) > 20:
            break

    df = pd.DataFrame(joined_data, columns=columns)
    df.to_csv(output_file)


if __name__ == "__main__":
    # for the annotations, we want to flatten these into the format:
    # array of: { id, image_id, image_filename, caption }
    training_captions, validation_captions = retrieve_captions()

    # combine the questions and the annotations files into:
    # array of: { question_id, question, image_id, question_type, answer_type, multiple_choice_answer, answer_id, answer }
    # note - there may be more than one entry for each question_id
    training_annotations, validation_annotations = retrieve_annotations()

    # now we can flatten these to combine them on the images with the captions
    # and output a CSV file for each
    combine_and_output(training_captions, training_annotations, TRAIN_OUTPUT_FILE)
    combine_and_output(
        validation_captions, validation_annotations, VALIDATION_OUTPUT_FILE
    )
