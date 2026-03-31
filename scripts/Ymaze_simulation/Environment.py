try:
    import cupy as np
    a=np.array([0])
except:
    import numpy as np
try:
    from Env_utils import position_expansion
except:
    from .Env_utils import position_expansion

class YMaze():
    """
    Mouse 2AFC navigation maze
    """
    def __init__(self,delay=None):
        """Maze setups, session-specific

        Args:
            delay (scalar, optional): Specify the cue offset (delay onset) position,
                if the task has a working memory period. Cue on during posF=[0,delay].
        """
        self.dtData = 0.186
        self.dt = 0.0093
        if delay is None:
            self.with_delay=False
        else:
            self.with_delay=True
            self.delay=delay
        #convert forward velocity to forward position offset
        self.posF_ratio = 18.96
        self.endmaze = 233
        #convert lateral velocity to lateral position offset
        #Args: x (scalar): lateral velocity
        self.posL_func=lambda x: 3.6 * np.tanh(6 * x)

    def _maze_restriction(self,posF,posL):
        """restrictions that define the Y-maze structure
        all positions are in unit cm
        """
        # forward position
        posF = np.clip(posF, 0, 240)
        # lateral position
        # 1/4 maze stem
        if posF <= 150:
            posL = np.clip(posL, -5, 5)
        # 2/4 maze arm
        posL = np.clip(posL, -35, 35)
        # 3/4 cannot cross the middle wall after 220cm
        if posF >= 220:
            if self.posL > 0 and posL < 5:
                posL = np.array(5)
            if self.posL < 0 and posL > -5:
                posL = np.array(-5)
        # 4/4 sphere tip of the middle wall (posF 215-220cm)
        # if the next location is inside the sphere, project it onto the sphere surface
        # this allows some last minute changes of the choice, which do exist in the data
        if posF < 220 and (posF - 220) ** 2 + posL ** 2 <= 25:
            c = np.sqrt((posF - 220) ** 2 + posL ** 2)
            posL = 5 / c * posL
            b = 5 / c * (220 - posF)
            posF = 220 - b
        return posF,posL

    def maze_init(self,cue,velF_init,velL_init,velY_init,posF_init,posL_init):
        """Initialize the mouse's state in the maze for each trial.

        Args:
            cue (-1 or 1): visual cue type, -1 indicates left 1 right
            velF_init (scalar): initial forward velocity
            velL_init (scalar): initial lateral velocity
            velL_init (scalar): initial yaw velocity
            posF_init (scalar): initial forward position
            posL_init (scalar): initial lateral position
        """
        self.cue = cue
        self.velF = velF_init
        self.velL = velL_init
        self.velY = velY_init
        self.posF = posF_init
        self.posL = posL_init
        self.posF, self.posL = self._maze_restriction(self.posF,self.posL)
        self._cue_update()

    def _cue_update(self):
        """return env input for the visual cue
        """
        self.cue_t = np.zeros(2)
        if self.posF <= self.delay:
            if self.cue == 1:
                self.cue_t[0] = 1
            elif self.cue == 0:
                self.cue_t[1] = 1

    def beh_update(self, velF, velL,velY):
        """Take in current velocity, update current position from past position

        Args:
            velF, velL (scalar): velocity at the current time step

        """
        posL = self.posL + self.posL_func(0.5 * (velL + self.velL)) * self.dt / self.dtData
        posF = self.posF + 0.5 * (velF + self.velF) * self.posF_ratio * self.dt / self.dtData
        #apply position restrictions
        posF,posL = self._maze_restriction(posF,posL)

        self.velF=velF
        self.velL=velL
        self.velY=velY
        self.posF=posF
        self.posL=posL
        self._cue_update()

    def beh_readout(self):
        return np.array([self.velF,self.velL,self.velY,self.posF,self.posL])

    def observations_from_env(self):
        beh_fb=np.zeros(9)
        beh_fb[0] = self.velF
        beh_fb[1] = self.velL
        beh_fb[2] = self.velY
        beh_fb[3:8] = position_expansion(np.array([self.posF]), 5, 0, 240)[0]
        beh_fb[8] = self.posL/35

        return np.hstack((self.cue_t, beh_fb)), self.posF >= self.endmaze


class Ymaze_obstacle(YMaze):
    def __init__(self,delay=None,obstacles=None):
        """obstacles: list of [posF, posL_left, posL_right], where the obstacle is located at 
        forward position posF, and occupies the lateral space between posL_left and posL_right
        """
        super().__init__(delay)
        self.obstacle=obstacles
        
    def maze_init(self,cue,velF_init,velL_init,velY_init,posF_init,posL_init):
        super().maze_init(cue,velF_init,velL_init,velY_init,posF_init,posL_init)
        self.gameover=False
        
    def _maze_restriction(self,posF,posL):
        """additional restriction of obstacles in the maze, on top of the Y-maze structure
        """
        posF,posL=super()._maze_restriction(posF,posL)
        for obstacle in self.obstacle:
            if self.posF<=obstacle[0] and posF>obstacle[0] and self.posL>=obstacle[1] and self.posL<=obstacle[2]:
                posF=np.array(obstacle[0])
                self.gameover=True
        return posF,posL
    
    def observations_from_env(self):
        obs,termination=super().observations_from_env()
        return obs,termination, self.gameover
    
    def observations_position(self):
        obs=np.zeros(10)
        obs[:5] = position_expansion(np.array([self.posF]), 5, 0, 240)[0]
        obs[5:] = position_expansion(np.array([self.posL]), 5, -35,35)[0]
        return obs
