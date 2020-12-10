PRCO305_audio_classification

<br/>
To add a new genre:<br/>
1: You will need a sufficient number of songs to use for regenerating the model, 10+ should be okay.
<br/>
2: Add the name of the genre to the files:
    <br/>
    A: utils/recorded_data_processing.py
    <br/>
    B: utils/data_processing_utils.py
    <br/>
    
3: Increment the number of TensorFlow nodes in the final layer (line 45 in data_training.py)<br/>
4: Run the file audio_clipping.py to rename and reformat each file into the correct types.
<br/>
5: Run the file data_processing_utils.py to update processed_data/data.csv
<br/>
6: run data_training.py to update the TensorFlow model in models for use.
