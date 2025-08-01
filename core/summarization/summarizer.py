from transformers import pipeline, AutoTokenizer
import gradio as gr


model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
summarizer_pipeline = pipeline("summarization", model=model_path)





def text_to_chunks(text, tokenizer, max_tokens=900):  # keep it below 1024 to be safe
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

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


def summarize_by_chunks(text, summarize_fn, max_tokens=900):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    chunks = text_to_chunks(text, tokenizer, max_tokens)
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            print(
                f"Summarizing chunk {i+1}/{len(chunks)} (approx {len(tokenizer.encode(chunk))} tokens)"
            )
            summary = summarize_fn(chunk)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")
    return " ".join(summaries)


# def summarize(input):
#     output = summarizer_pipeline(input)
#     return output[0]["summary_text"]
# def summarizer(input):
#     output = summarize_by_chunks(input, summarize, 900)
#     return output


def summarizer(text):
    return summarize_by_chunks(text, summarizer_pipeline, 900)


# def large_summarizer(long_text):
#     summary = summarizer_pipeline(
#         long_text,
#         max_length=50,
#         min_length=10,
#         do_sample=False,
#     )
#     print(f"Summary: {summary[0]['summary_text']}")


def main():

    #     text = """
    #             The axolotl (/ˈæksəlɒtəl/ ⓘ; from Classical Nahuatl: āxōlōtl [aːˈʃoːloːtɬ] ⓘ) (Ambystoma mexicanum) is a paedomorphic salamander, one that matures without undergoing metamorphosis into the terrestrial adult form; adults remain fully aquatic with obvious external gills. This trait is somewhat unusual among amphibians, though this trait is not unique to axolotls, and this is apparent as they may be confused with the larval stage or other neotenic adult mole salamanders (Ambystoma spp.), such as the occasionally paedomorphic tiger salamander (A. tigrinum) widespread in North America; or with mudpuppies (Necturus spp.), which bear a superficial resemblance but are from a different family of salamanders.[4]
    #             Axolotls originally inhabited a system of interconnected wetlands and lakes in the Mexican highlands; they were known to inhabit the smaller lakes of Xochimilco and Chalco, and are also presumed to have inhabited the larger lakes of Texcoco and Zumpango. These waterways were mostly drained by Spanish settlers after the conquest of the Aztec Empire, leading to the destruction of much of the axolotl's natural habitat, which is now largely occupied by Mexico City. Despite this, they remained abundant enough to form part of the staple in the diet of native Mexica during the colonial era.[5] Due to continued urbanization in Mexico City, which causes water pollution in the remaining waterways, as well as the introduction of invasive species such as tilapia and carp, the axolotl is near extinction, the species being listed as critically endangered in the wild, with a decreasing population of around 50 to 1,000 adult individuals, by the International Union for Conservation of Nature (IUCN) and is listed under Appendix II of the Convention on International Trade in Endangered Species (CITES).[2]
    #             A large captive population of axolotls currently exist, with the specimens being used extensively in scientific research for their remarkable ability to regenerate parts of their body, including limbs, gills and parts of their eyes and brains. In general, they are model organisms that are also used in other research matters, and as aquarium technology developed, they have become a common exhibit in zoos and aquariums, and as an occasional pet in home aquaria. Axolotls are also a popular subject in contemporary culture, inspiring a number of works and characters in media.
    #             A sexually mature adult axolotl, at age 18–27 months, ranges in length from 15 to 45 cm (6 to 18 in), although a size close to 23 cm (9 in) is most common and greater than 30 cm (12 in) is rare. Axolotls possess features typical of salamander larvae, including external gills and a caudal fin extending from behind the head to the vent.[11][12] External gills are usually lost when salamander species mature into adulthood, although the axolotl maintains this feature.[13] This is due to their neoteny, where axolotls are much more aquatic than other salamander species.[14] Their heads are wide, and their eyes are lidless. Their limbs are underdeveloped and possess long, thin digits. Three pairs of external gill stalks (rami) originate behind their heads and are used to move oxygenated water. The external gill rami are lined with filaments (fimbriae) to increase surface area for gas exchange.[13] Four-gill slits lined with gill rakers are hidden underneath the external gills, which prevent food from entering and allow particles to filter through. Males can be identified by their swollen cloacae lined with papillae, while females have noticeably wider bodies when gravid and full of eggs.
    #             Buccal pumping
    #             Axolotls have barely visible vestigial teeth, which develop during metamorphosis. The primary method of feeding is by suction, during which their rakers interlock to close the gill slits. External gills are used for respiration, although buccal pumping (gulping air from the surface) may also be used to provide oxygen to their lungs.[13] Buccal pumping can occur in a two-stroke manner that pumps air from the mouth to the lungs, and with four-stroke that reverses this pathway with compression forces.
    #             Captive axolotl color morphs
    #             The wild type animal (the "natural" form) is brown or tan with gold speckles and an olive undertone, and possess an ability to subtly alter their color by changing the relative size and thickness of their melanophores, presumably for camouflage.[15] Axolotls have four pigmentation genes; when mutated, they create different color variants.[citation needed] The five most common mutant colors are listed below;[clarification needed]
    #             Leucistic: pale pink with black eyes.
    #             Xanthic: grey, with black eyes.
    #             Albinism: pale pink or white, with red eyes.
    #             Melanism: all black or dark blue with no gold speckling or olive tone.
    #             In addition, there is wide individual variability in the size, frequency, and intensity of the gold speckling, and at least one variant develops a black and white piebald appearance upon reaching maturity.[16] Because pet breeders frequently cross the variant colors, double homozygous mutants are common in the pet trade, especially white/pink animals with pink eyes that are double homozygous mutants for both the albino and leucistic genes.[17]
    #             Melanophores of a larva axolotl
    #             The 32 billion base pair long sequence of the axolotl's genome was published in 2018 and was the largest animal genome completed at the time. It revealed species-specific genetic pathways that may be responsible for limb regeneration.[18] Although the axolotl genome is about 10 times as large as the human genome, it encodes a similar number of proteins, namely 23,251[18] (the human genome encodes about 20,000 proteins). The size difference is mostly explained by a large fraction of repetitive sequences, but such repeated elements also contribute to increased median intron sizes (22,759 bp) which are 13, 16 and 25 times that observed in human (1,750 bp), mouse (1,469 bp) and Tibetan frog (906 bp), respectively.[18]
    #             Physiology
    #             Regeneration
    #             The feature of the axolotl that attracts most attention is its healing ability: the axolotl does not heal by scarring, but is capable of tissue regeneration; entire lost appendages such as limbs and the tail are regrow over a period of months, and, in certain cases, more vital structures, such as the tissues of the eye and heart can be regrown.[19][20] They can restore parts of their central nervous system, such as less vital parts of their brains. They can also readily accept transplants from other individuals, including eyes and parts of the brain—restoring these alien organs to full functionality. In some cases, axolotls have been known to repair a damaged limb, as well as regenerating an additional one, ending up with an extra appendage that makes them attractive to pet owners as a novelty. Their ability to regenerate declines with age but does not disappear, though in metamorphosed individuals, the ability to regenerate is greatly diminished. Axolotls experience indeterminate growth, their bodies continuing to grow throughout their life, and some consider this trait to be a direct contributor to their regenerative abilities.[21] The axolotl is therefore used as a model for the development of limbs in vertebrates.[22] There are three basic requirements for regeneration of the limb: the wound epithelium, nerve signaling, and the presence of cells from the different limb axes.[23] A wound epidermis is quickly formed by the cells to cover up the site of the wound. In the following days, the cells of the wound epidermis divide and grow, quickly forming a blastema, which means the wound is ready to heal and undergo patterning to form the new limb.
    #             It is believed that during limb generation, axolotls have a different system to regulate their internal macrophage level and suppress inflammation, as scarring prevents proper healing and regeneration.[24] However, this belief has been questioned by other studies.[25] The axolotl's regenerative properties leave the species as the perfect model to study the process of stem cells and its own neoteny feature. Current research can record specific examples of these regenerative properties through tracking cell fates and behaviors, lineage tracing skin triploid cell grafts, pigmentation imaging, electroporation, tissue clearing and lineage tracing from dye labeling. The newer technologies of germline modification and transgenesis are better suited for live imaging the regenerative processes that occur for axolotls.[26]
    #             """
    #
    # result = summarize_chunks(text, summarize, 900)
    # result = large_summarizer(text)

    # result = summarizer_interface(text)
    # print(result)

    gr.close_all()

    interface = gr.Interface(
        fn=summarizer,
        inputs=gr.Textbox(label="Enter text to summarize", lines=10),
        outputs=gr.Textbox(label="Summary", lines=5),
        title="Summarizer",
        description="Summarizes a gives text corpus",
    )
    interface.launch()


if __name__ == "__main__":
    main()
