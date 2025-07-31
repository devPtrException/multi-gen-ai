import random
import json
import gradio as gr
from transformers import pipeline, AutoTokenizer

model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
translator_pipeline = pipeline("translation", model=model_path, max_length=400)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def get_FLORES_code_from_language(language):
    with open("core/translator/languages.json", "r") as file:
        language_data = json.load(file)

    for entry in language_data:
        if entry["Language"].lower() == language.lower().strip():
            return entry["FLORES-200 code"]
    return None


def text_to_chunks(text, tokenizer, max_tokens=180):
    words = text.split()
    chunks, current_chunk, current_len = [], [], 0

    for word in words:
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        if current_len + len(word_tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = len(word_tokens)
        else:
            current_chunk.append(word)
            current_len += len(word_tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def translate_by_chunks(text, src_lang, tgt_lang, pipeline, tokenizer, max_tokens=180):
    chunks = text_to_chunks(text, tokenizer, max_tokens)
    translations = []
    src_code = get_FLORES_code_from_language(src_lang)
    tgt_code = get_FLORES_code_from_language(tgt_lang)

    if not src_code or not tgt_code:
        return f"Invalid source/target language. Check supported codes."

    for i, chunk in enumerate(chunks):
        try:
            print(
                f"Translating chunk {i+1}/{len(chunks)} ({len(tokenizer.encode(chunk))} tokens)"
            )
            translated = pipeline(chunk, src_lang=src_code, tgt_lang=tgt_code)
            translations.append(translated[0]["translation_text"])
        except Exception as e:
            print(f"Error in chunk {i}: {e}")

    return " ".join(translations)


def get_languages():

    with open("core/translator/languages.json", "r") as file:
        language_data = json.load(file)

    languages = []
    for entry in language_data:
        languages.append(entry["Language"])

    #     for language in languages:
    #         print(language)
    #
    #     print(languages)

    return languages


def translator_interface(text, source_lang, target_lang):
    if not text.strip():
        return "Please enter some text to translate"

    return translate_by_chunks(
        text, source_lang, target_lang, translator_pipeline, tokenizer
    )


def swap_languages(src, tgt):
    return tgt, src


def main():
    #     text = """
    #         The axolotl (/ˈæksəlɒtəl/ ⓘ; from Classical Nahuatl: āxōlōtl [aːˈʃoːloːtɬ] ⓘ) (Ambystoma mexicanum) is a paedomorphic salamander, one that matures without undergoing metamorphosis into the terrestrial adult form; adults remain fully aquatic with obvious external gills. This trait is somewhat unusual among amphibians, though this trait is not unique to axolotls, and this is apparent as they may be confused with the larval stage or other neotenic adult mole salamanders (Ambystoma spp.), such as the occasionally paedomorphic tiger salamander (A. tigrinum) widespread in North America; or with mudpuppies (Necturus spp.), which bear a superficial resemblance but are from a different family of salamanders.[4]
    #         Axolotls originally inhabited a system of interconnected wetlands and lakes in the Mexican highlands; they were known to inhabit the smaller lakes of Xochimilco and Chalco, and are also presumed to have inhabited the larger lakes of Texcoco and Zumpango. These waterways were mostly drained by Spanish settlers after the conquest of the Aztec Empire, leading to the destruction of much of the axolotl's natural habitat, which is now largely occupied by Mexico City. Despite this, they remained abundant enough to form part of the staple in the diet of native Mexica during the colonial era.[5] Due to continued urbanization in Mexico City, which causes water pollution in the remaining waterways, as well as the introduction of invasive species such as tilapia and carp, the axolotl is near extinction, the species being listed as critically endangered in the wild, with a decreasing population of around 50 to 1,000 adult individuals, by the International Union for Conservation of Nature (IUCN) and is listed under Appendix II of the Convention on International Trade in Endangered Species (CITES).[2]
    #         A large captive population of axolotls currently exist, with the specimens being used extensively in scientific research for their remarkable ability to regenerate parts of their body, including limbs, gills and parts of their eyes and brains. In general, they are model organisms that are also used in other research matters, and as aquarium technology developed, they have become a common exhibit in zoos and aquariums, and as an occasional pet in home aquaria. Axolotls are also a popular subject in contemporary culture, inspiring a number of works and characters in media.
    #         A sexually mature adult axolotl, at age 18–27 months, ranges in length from 15 to 45 cm (6 to 18 in), although a size close to 23 cm (9 in) is most common and greater than 30 cm (12 in) is rare. Axolotls possess features typical of salamander larvae, including external gills and a caudal fin extending from behind the head to the vent.[11][12] External gills are usually lost when salamander species mature into adulthood, although the axolotl maintains this feature.[13] This is due to their neoteny, where axolotls are much more aquatic than other salamander species.[14] Their heads are wide, and their eyes are lidless. Their limbs are underdeveloped and possess long, thin digits. Three pairs of external gill stalks (rami) originate behind their heads and are used to move oxygenated water. The external gill rami are lined with filaments (fimbriae) to increase surface area for gas exchange.[13] Four-gill slits lined with gill rakers are hidden underneath the external gills, which prevent food from entering and allow particles to filter through. Males can be identified by their swollen cloacae lined with papillae, while females have noticeably wider bodies when gravid and full of eggs.
    #         Buccal pumping
    #         Axolotls have barely visible vestigial teeth, which develop during metamorphosis. The primary method of feeding is by suction, during which their rakers interlock to close the gill slits. External gills are used for respiration, although buccal pumping (gulping air from the surface) may also be used to provide oxygen to their lungs.[13] Buccal pumping can occur in a two-stroke manner that pumps air from the mouth to the lungs, and with four-stroke that reverses this pathway with compression forces.
    #         """
    #
    #     result = translator_interface(text, "English", "French")
    #     print(result)

    # gr.close_all()
    # interface = gr.Interface(
    #     fn=translator_interface,
    #     inputs=[
    #         gr.Textbox(label="Enter text to translate:", lines=8),
    #         gr.Dropdown(
    #             get_languages(),
    #             label="Select Target Language:",
    #         ),
    #         gr.Dropdown(
    #             get_languages(),
    #             label="Select Target Language:",
    #         ),
    #     ],
    #     outputs=gr.Textbox(label="Translated Text", lines=6),
    #     title="Chunk-Based Multi Linguistic Translator (NLLB-200)",
    #     description="Translates long texts using Facebook's NLLB-200-distilled model...",
    # )
    # interface.launch()

    languages = get_languages()

    # Pick a random language that’s not the same as the default source
    default_src = languages[0]
    other_langs = [lang for lang in languages if lang != default_src]
    default_tgt = random.choice(other_langs) if other_langs else default_src

    with gr.Blocks(
        css="""
        #swap-label {
            font-size: 12px;
            line-height: 1.5;
            padding: 4px 0;
            margin: 0;
            text-align: center;
            height: 30px;
        }

        #swap-btn {
            font-weight: bold;
            height: 40px;
            margin-top: 4px;
        }

        .swap-col {
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 60px;
        }
    """
        #         css="""
        #             #swap-label {
        #             font-size: 12px;
        #             line-height: 1;
        #             padding: 0;
        #             margin: 0;
        #             text-align: center;
        #             height: 20px;
        #         }
        #
        #         #swap-btn {
        #             font-weight: bold;
        #             height: 40px;
        #             margin-top: 4px;
        #         }
        #
        #         /* Match height of dropdown column */
        #         .swap-col {
        #             display: flex;
        #             flex-direction: column;
        #             justify-content: center;
        #         }
        #         """
    ) as interface:
        gr.Markdown("## Chunk-Based Multi Linguistic Translator (NLLB-200)")

        with gr.Row():
            input_text = gr.Textbox(label="Enter text to translate:", lines=8)

        with gr.Row():
            with gr.Column(scale=1, elem_id="source-col"):
                source_lang = gr.Dropdown(
                    choices=languages,
                    label="Source Language (scroll/type to select)",
                    value=languages[0],
                    filterable=True,
                )

            with gr.Column(scale=0.7, elem_classes="swap-col"):
                gr.Label(
                    "Swap source and target languages",
                    show_label=False,
                    elem_id="swap-label",
                )
                swap_button = gr.Button("🔄 Swap", elem_id="swap-btn")

            with gr.Column(scale=1, elem_id="target-col"):
                target_lang = gr.Dropdown(
                    choices=languages,
                    label="Target Language (scroll/type to select)",
                    value=default_tgt,
                    filterable=True,
                )

            swap_button.click(
                fn=swap_languages,
                inputs=[source_lang, target_lang],
                outputs=[source_lang, target_lang],
            )

        output_text = gr.Textbox(label="Translated Text", lines=6)

        translate_button = gr.Button("Translate")
        translate_button.click(
            fn=translator_interface,
            inputs=[input_text, source_lang, target_lang],
            outputs=output_text,
        )

    interface.launch()


if __name__ == "__main__":
    main()
