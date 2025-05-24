![bayes_beats_banner](https://github.com/user-attachments/assets/6047ab36-561f-4652-95f7-b51c4c86b148)
Welcome to Bayes Beats, a Bayesian-powered music generator that creates sick beats tailored to YOUR mood 🤯

* Pumped for the end of the semester? We have you covered 🎉
* Deep in vibe coding mode? Also gotchu 🤓
* Going through a breakup? 😭 You deserve better and we're here for all the feels

Need something more specific? Maybe you're thinking "hike through the Himalayas while contemplating the existence of flatware?" Weird, but we can handle that 💪

So what are you waiting for? 

👇 **Click an image to get started** 👇


**Methodology**

Our foundational model is [ACE-STEP](https://github.com/ace-step/ACE-Step?tab=readme-ov-file#-features). ACE-Step is a model built to generate music from text inputs. It is a diffusion model but is complemented by a 
deep-compression auto-encoder and linear transformer. We modify the 

We fine tune this model on MTG-JAMENDO, a large dataset of annotated tracks that works has mood-specific labels. The goal of fine-tuning is to train ACE-STEP to be atuned to capturing "moods" in the songs it outputs. We use Low Rank Approximation methods, and only fine-tune the weights associated with the last genreative layer.  

