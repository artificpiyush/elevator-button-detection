class FloorSelector:
    """
    Baseline exact-match floor selector.
    """
    @staticmethod
    def find_target_button(buttons, target_floor):
        """
        Finds the button exactly matching the target floor string.
        """
        if target_floor is None:
            return None
            
        target_str = str(target_floor).strip().lower()
        
        for btn in buttons:
            if btn.get('text', '').strip().lower() == target_str:
                return btn
                
        return None
