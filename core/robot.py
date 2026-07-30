from core.state import RobotState

class Robot:

    def __init__(self):
        self.state = RobotState.IDLE

    def set_state(self, state: RobotState):
        self.state = state
        print(f"🤖 State -> {state.value}")