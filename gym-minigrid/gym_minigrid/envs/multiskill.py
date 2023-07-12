from gym.utils import seeding
import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import itertools as itt

SKILLS = [
    'SimpleCrossing',
    'LavaCrossing',
    'DoorKey',
]

class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos,
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos


class MultiSkillEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        seed=1337,
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []

        self.skills = set(SKILLS)

        super().__init__(
            grid_size=4 + (self.maxNumRooms - 1) * (self.maxRoomSize - 2) + self.maxRoomSize - 1,
            max_steps=self.maxNumRooms * 4 * maxRoomSize * maxRoomSize,
            seed=seed,
        )

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # reset full list of skills
        self.skills = set(SKILLS)

        # Return first observation
        obs = self.gen_obs()
        return obs

    def _gen_grid(self, width, height):

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        # Debug
        idx = 0

        while len(roomList) < numRooms:
            curRoomList = []

            if idx == 0:
                entryDoorPos = (2, 2)
            else:
                entryDoorPos = (
                    self._rand_int(0, width - 2),
                    self._rand_int(0, width - 2)
                )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=self.maxRoomSize, # 4,
                maxSz=self.maxRoomSize,
                entryDoorWall=4,  # 2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

            idx += 1

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                # instead of creating a coloured door which has not been seen during training
                # fo individual tasks SC, DK, LC, let's just create a gap
                #self.grid.set(*room.entryDoorPos, entryDoor)
                self.grid.set(*room.entryDoorPos, None)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

            # If this isn't the first room or last room
            # Let's add a crossing task
            if idx > 0 and idx < len(roomList)-1:
                # Place obstacles (lava or walls)
                # Again Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                skill = self._rand_elem(sorted(self.skills))

                if skill == 'DoorKey':
                    # remove DoorKey from set of skills
                    # the same key can be used for multiple doors
                    self.skills.remove(skill)

                    # Create a vertical splitting wall
                    splitIdx = self._rand_int(topX + 2, topX + sizeX - 2)
                    self.grid.vert_wall(splitIdx, topY, sizeY) # x, y, length

                    # Place a door in the wall
                    doorIdx = self._rand_int(topY + 1, topY + sizeY - 1)
                    self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

                    # Place a yellow key on the left side
                    self.place_obj(
                        obj=Key('yellow'),
                        top=(topX, topY), # top left position
                        size=(splitIdx - topX, sizeY) # size of the rectanle
                    )

                else:

                    self.obstacle_type = Lava if 'Lava' in skill else Wall

                    assert sizeX % 2 == 1 and sizeY % 2 == 1  # odd size

                    num_crossings = 1

                    # Place obstacles (lava or walls)
                    v, h = object(), object()  # singleton `vertical` and `horizontal` objects

                    # Lava rivers or walls specified by direction and position in grid
                    rivers = [(v, i) for i in range(topX + 2, topX + sizeX - 2, 2)]
                    rivers += [(h, j) for j in range(topY + 2, topY + sizeY - 2, 2)]
                    self.np_random.shuffle(rivers)
                    rivers = rivers[:num_crossings]  # sample random rivers

                    rivers_v = sorted([pos for direction, pos in rivers if direction is v])
                    rivers_h = sorted([pos for direction, pos in rivers if direction is h])
                    obstacle_pos = itt.chain(
                        itt.product(range(topX + 1, topX + sizeX - 1), rivers_h),
                        itt.product(rivers_v, range(topY + 1, topY + sizeY - 1)),
                    )

                    for i, j in obstacle_pos:
                        self.put_obj(self.obstacle_type(), i, j)

                    # Sample path to goal
                    path = [h] * len(rivers_v) + [v] * len(rivers_h)
                    self.np_random.shuffle(path)

                    # Create openings
                    limits_v = [topX] + rivers_v + [topX + sizeY - 1]
                    limits_h = [topY] + rivers_h + [topY + sizeX - 1]
                    room_i, room_j = 0, 0
                    for direction in path:
                        if direction is h:
                            i = limits_v[room_i + 1]
                            j = self.np_random.choice(
                                range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                            room_i += 1
                        elif direction is v:
                            i = self.np_random.choice(
                                range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                            j = limits_h[room_j + 1]
                            room_j += 1
                        else:
                            assert False
                        self.grid.set(i, j, None)

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = 9
        sizeY = 9

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        elif entryDoorWall == 4:
            topX = entryDoorPos[0]
            topY = entryDoorPos[1] - 1
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
            )
        )

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            if entryDoorWall == 4:
                exitDoorWall = 4
                nextEntryWall = 4
            else:
                wallSet = set((0, 1, 2, 3))
                wallSet.remove(entryDoorWall)
                exitDoorWall = self._rand_elem(sorted(wallSet))
                nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            elif exitDoorWall == 4:
                exitDoorPos = (topX + sizeX - 1, topY + sizeY - 2)
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

class MultiSkillEnvN2(MultiSkillEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2+2, # starting room end room + 2 kill rooms
            maxNumRooms=2+2,
            seed=1337,
        )

class MultiSkillEnvN3(MultiSkillEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=3+2, # starting room end room + 2 kill rooms
            maxNumRooms=3+2,
            seed=1337,
        )

class MultiSkillEnvN4(MultiSkillEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=4+2, # starting room end room + 2 kill rooms
            maxNumRooms=4+2,
            seed=1337,
        )

register(
    id='MiniGrid-MultiSkill-N2-v0',
    entry_point='gym_minigrid.envs:MultiSkillEnvN2'
)

register(
    id='MiniGrid-MultiSkill-N3-v0',
    entry_point='gym_minigrid.envs:MultiSkillEnvN3'
)

register(
    id='MiniGrid-MultiSkill-N4-v0',
    entry_point='gym_minigrid.envs:MultiSkillEnvN4'
)

