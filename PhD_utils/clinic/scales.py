

def get_crsr(auditory_func: int,
            visual_func: int,
            motor_func: int,
            verb_scale: int,
            comm_scale: int,
            arousal: int):
    
    """ Brief: Calculate the Conscious state COMA, UWS, MCS-/+, EMCS
    """

    if motor_func == 6 or comm_scale == 2:
        return "EMCS"
    if auditory_func > 2 or visual_func == 5 or verb_scale > 2 or comm_scale == 1:
        return "MCS+"
    if visual_func > 1 or motor_func > 2:
        return "MCS-"
    if arousal == 0:
        return "COMA"

    return "UWS"
