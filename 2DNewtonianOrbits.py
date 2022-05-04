import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time


def night():
    t = time.localtime().tm_hour
    if t<7 or t>20:
        return True
    else:
        return False
if night():
    plt.style.use('dark_background')

class SolarSystem () :
    """ This class creates the SolarSystem object."""
    def __init__ (self, leapfrog = True):
        """ With self you can access private atributes of the
        object .
        """
        self.size = 1000
        self.planets = []
        # This initializes the 3D figure
        self.fig, self.ax = plt.subplots ()
        # self.ax = Axes3D ( self.fig , auto_add_to_figure = False ) #-3D
        # self.fig.add_axes ( self.ax ) #-3D
        self.total_stats = {'energy':[], 'angmo':[]}
        self.fconst = 8
        self.leapfrog = leapfrog
        self.dT = 1
        if self.leapfrog:
            self.dT *=1/2
        
    def add_planet ( self , planet ):
        """ Every time a planet is created it gets put into
        the array .
        """
        self.planets.append ( planet )
        
    def update_planets ( self ):
        """ This method moves and draws all of the planets."""
        self.ax.clear ()
        for planet in self.planets :
            planet.move ()

    
    @classmethod
    def overwrite_update_to_quick (cls, bool):
        """Overwrites update_planets method so that 
        self.ax.clear() and planet.draw will not be executed 
        and hence only planet.move() will be executed,
        when animation is not required."""
        if bool:
            def new_update_planets ( self ):
                """ This method moves and draws all of the planets."""
                # self.ax.clear ()
                for planet in self.planets :
                    planet.move ()
                    # planet.draw ()
            cls.update_planets = new_update_planets
            print("settled")
            
    def fix_axes ( self ):
        """ The axes would change with each iteration
        otherwise .
        """
        self.ax.set_xlim (( - self.size  , self.size) ) # -bigcanvas
        self.ax.set_ylim (( - self.size  , self.size) ) # -bigcanvas
        # self.ax.set_xlim (( - self.size /2 , self.size /2) )
        # self.ax.set_ylim (( - self.size /2 , self.size /2) )
        # self.ax.set_zlim (( - self.size /2 , self.size /2) ) #-3D
        
    def gravity_planets ( self ) :
        """ This method calculated gravity interaction for
        every planet .
        """
        # for i , first in enumerate ( self.planets ) :
        #     for second in self.planets [ i +1:]:
        #         first.gravity ( second )

        su      = self.get_planets("Sun")# -Sungravonly
        planets = self.get_planets("Planet")# -Sungravonly
        for p in planets:# -Sungravonly
            p.gravity(su)# -Sungravonly
    
    def get_planets (self, target):
        # target has to be in str form
        targets = [p for p in self.planets if type(p).__name__== target]
        if len(targets)==1 and target == "Sun":
            return targets[0]
        elif len(targets)!=0:
            return targets
        else:
            return None # optional, returns None by default

    # def save_status (self, stats_show = ["energy", "angmo"]):
    #     """stats_show: list of str(stats)"""
    #     su = self.get_planets("Sun")
    #     stats_show = [stat + "1f" if stat in ['position', 'velocity'] else stat for stat in stats_show]
    #     total_stats = {k:0 for k in stats_show if k not in ["position1f", "velocity1f"]}
    #     for p in self.planets:
    #         if p == su:
    #             continue
    #         stats = p.get_status()
    #         for s in total_stats.keys():
    #             total_stats[s] += stats[s]
    #     #     # display stats along with planets
    #     #     stats_tups       = [f"{k}: {v}" for (k, v) in stats.items() if (k in stats_show and type(v)==type(()))]
    #     #     stats_floats = [f"{k}: {v:.1f}" for (k, v) in stats.items() if (k in stats_show and isinstance(v, float))]
    #     #     self.ax.text(*p.position, "\n".join(stats_tups + stats_floats))
    #     for k, v in total_stats.items():
    #         self.total_stats[k].append(v)
    #     # # display total energy and total angular momentum at the top right corner
    #     # self.ax.text(self.size/5,
    #     #             self.size/3, 
    #     #             "\n".join([f"Total {k}: {v:.1f}" for (k,v) in total_stats.items() if k in ["energy", "angmo"]]))

    def after_draw(self, i, trail = True, calc_tot = False):
        """Draws the movement of the planets after simulating and storing all the frames"""
        self.ax.clear()
        # plot planets
        for p in self.planets:
            self.ax.plot(*p.positions[i], color = p.color, marker ="o", markersize =10)
            if trail:
                self.ax.plot(*[t for t in zip(*p.positions[:i])], color = p.color)

        # shows energy and momentum
        # if calc_tot:
        #     self.ax.text(self.size*0.6, 0.04, 
        #                 f"Total KE: {self.tot_kinetic[i]:.1f}\nTotal mo: {self.tot_momentum[i]:.1f}")
        self.fix_axes()
        # # self.ax.set_ylim(0, self.t + 10)
    
    def plot_trail(self, ax, color = None, planet = None, planet_positions = None):
        if planet != None:
            color = planet.color
            trail = planet.positions
        elif planet_positions != None:
            trail = planet_positions
        ax.plot(*[t for t in zip(*trail[:i])], color = color)

    def show_animation(self, f=700, playspeed = 4):
        # shows animation of simulation based on frames ran
        def animate(i):
            self.after_draw(i*playspeed, calc_tot = False)
        anim = animation.FuncAnimation(self.fig, animate, frames = int(f/playspeed), interval = 1)
        return anim

class Planet () :
    """ This class creates the Planet object."""
    def __init__ (  self ,
                    SolarSys ,
                    mass ,
                    position =(0 , 0, 0) ,
                    velocity =(0 , 0, 0) ,
                    color = "c" ):
        self.SolarSys = SolarSys
        self.mass = mass
        self.position = position
        self.velocity = velocity
        # The planet is automatically added to the SolarSys.
        self.SolarSys.add_planet ( self )
        self.color = color
        self.positions = []
        self.velocities = []
        
    def move ( self ) :
        """ The planet is moved based on the velocity."""
        self.positions.append(self.position)
        self.velocities.append(self.velocity)
        self.position = (
        self.position [0]+ self.velocity [0]* SolarSys.dT ,
        self.position [1]+ self.velocity [1]* SolarSys.dT #,
        # self.position [2]+ self.velocity [2]* SolarSys.dT #-3D
        )
        
    # def draw ( self ) :
        # """ The method to draw the planet."""
        # self.SolarSys.ax.plot (* self.position ,
        #                         marker ="o",
        #                         markersize =10 ,
        #                         color = self.color
        #                         )
        
    def gravity ( self , other ):
        """ The method to compute gravitational force for two
        planets.numpy module is used to handle vectors .
        """
        distance = np.subtract ( other.position , self.position )
        distanceMag = np.linalg.norm ( distance )
        distanceUnit = np.divide ( distance , distanceMag )
        forceMag = self.SolarSys.fconst * self.mass * other.mass / ( distanceMag **2)
        # forceMag = 1 * ( distanceMag ** 1)
        force = np.multiply ( distanceUnit , forceMag )
        # Switch makes force on self opossite to other
        switch = 1
        if self.SolarSys.leapfrog:
            switch *= 2
        for body in self , other :
            acceleration = np.divide ( force , body.mass )
            acceleration = np.multiply ( acceleration , SolarSys.dT * switch )
            # acceleration = np.multiply ( acceleration , SolarSys.dT * switch * 2) # -Leapfrog
            body.velocity = np.add ( body.velocity ,
            acceleration )
            switch *= -1
    
    def get_status (self):
        """Return a dict of relevant values of the planet (e.g. Energy, Angular Momentum)"""
        su = self.SolarSys.get_planets("Sun")
        xabs = np.linalg.norm(self.position)
        vabs = np.linalg.norm(self.velocity)
        
        # E=m(v^2)/2 - GMm/(r^2)
        energy = self.mass*(vabs**2)/2 - self.SolarSys.fconst*su.mass*self.mass/(xabs)
        # J = mrxv = r_x*v_y - r_y*v_x
        r = np.subtract(self.position, su.position)
        angmo = self.mass * np.cross(r, self.velocity)
        return {"position": self.position, 
                "velocity": self.velocity, 
                "position1f" : tuple(round(p,1) for p in self.position),
                "velocity1f" : tuple(round(v,1) for v in self.velocity),
                "energy": energy, 
                "angmo": angmo}

class Sun ( Planet ):
    """ This class is inherited from Planet.Everything is
    the same as in planet , except that the position of the
    sun is fixed.Also , the color is yellow .
    """
    def __init__ (
                self ,
                SolarSys ,
                mass =1000 ,
                position =(0 , 0) ,
                velocity =(0 , 0)
                ) :
        super ( Sun , self ). __init__ ( SolarSys , mass , position , velocity )
        self.positions = []
        self.color = "y"
        
    def move ( self ) :
        self.positions.append(self.position)
        self.position = self.position
        
def initSolarSun(leapfrog = True):
    # initialises the Solar System and the Sun
    SolarSys = SolarSystem (leapfrog = leapfrog)
    sun      = Sun (SolarSys)
    return (SolarSys, sun)

def circular_v(SolarSys, sun, px, py):
    return np.sqrt(SolarSys.fconst*sun.mass/np.sqrt(px**2+py**2))

###_______________________________________________________
### 1. Euler vs Leapfrog:
# one = input("Euler vs Leapfrog: (press Enter to view)")
# if one == "":
if True:
    ### 1.1 Euler
    SolarSys, sun = initSolarSun(leapfrog = False)

    # Instantiating of planets.
    px, py = 100, 4
    # v: for circular orbit at a given location
    v = circular_v(SolarSys, sun, px, py)
    theta = np.arctan(py/px)

    planet0 = Planet ( SolarSys ,
                        mass =10 ,
                        position =(px , py) ,
                        velocity =(-v*np.sin(theta) , v*np.cos(theta))
                        )

    f = 500
    print("Running simulation... this could take a couple of minutes")
    for i in range(f):
        SolarSys.gravity_planets ()
        SolarSys.update_planets ()
        
        SolarSys.fix_axes()

    animate = True
    # animate = False
    playspeed = int(10/2)

    if animate:
        anim = SolarSys.show_animation(f = f, playspeed=playspeed)
        plt.show()

    Eulerpositions = planet0.positions
    Eulervelocities = planet0.velocities

    ### 1.2 Leapfrog
    SolarSys, sun = initSolarSun(leapfrog = True) #(leapfrog mode will automatically be true for all other simulations below)
    # Instantiating of planets.
    px, py = 100, 4
    # v: for circular orbit at a given location
    v = circular_v(SolarSys, sun, px, py)
    theta = np.arctan(py/px)

    planet0 = Planet ( SolarSys ,
                        mass =10 ,
                        position =(px , py) ,
                        velocity =(-v*np.sin(theta) , v*np.cos(theta)),
                        )

    # Leapfrog
    f *= 2
    playspeed = 10
    print("Running simulation... this could take a couple of minutes")
    for i in range(f):
        SolarSys.update_planets () # -Leapfrog
        SolarSys.gravity_planets ()
        SolarSys.update_planets ()
        
        SolarSys.fix_axes()

    animate = True
    # animate = False

    planet0.color = "w"
    if animate:
        anim = SolarSys.show_animation(f = f, playspeed=playspeed)
        plt.show()

    Leapfrogpositions = planet0.positions
    Leapfrogvelocities = planet0.velocities



    # comparing trails of Euler and Leapfrog methods against circle
    fig, (ax1, ax2) = plt.subplots(1,2)
    # Euler and Leapfrog
    SolarSys.plot_trail(ax1, "c", planet_positions=Eulerpositions)
    SolarSys.plot_trail(ax2, "w", planet_positions=Leapfrogpositions)
    # perfect circle
    theta = np.linspace(0, 6.5, 100)
    r = (px**2 + py**2)**(1/2)
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    ax1.plot(x,y, color = "g", ls = ":")
    ax2.plot(x,y, color = "g", ls = ":")
    ax1.set_title("Euler: cyan; perfect circle: green")
    ax2.set_title("Leapfrog: white; perfect circle: green")
    txt = """
    The initial conditions were set up such that a circular orbit would be expected.\n
    Leapfrog shows a closer match to a circular orbit compared to the Euler method, due to smaller iteration intervals.
    """
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center')
    plt.show()

    # comparing Euler energies and Leapfrog's against time
    fig, (ax1, ax2) = plt.subplots(1,2)
    xabs1 = np.array([np.linalg.norm(p) for p in Eulerpositions])
    xabs2 = np.array([np.linalg.norm(p) for p in Leapfrogpositions])[::2]
    xabs2 = xabs2[:len(xabs2)//2]
    vabs1 = np.array([np.linalg.norm(v) for v in Eulervelocities])
    vabs2 = np.array([np.linalg.norm(v) for v in Leapfrogvelocities])[::2]
    vabs2 = vabs2[:len(vabs2)//2]
    # leapfrog method has 4 times more data (record twice per iteration, ran twice as many frames)
    # [::2] selects only one of the two records per iterations, and the list is sliced into half again

    # E=m(v^2)/2 - GMm/(r^2)
    energies1 = planet0.mass*(vabs1**2)/2 - planet0.SolarSys.fconst*sun.mass*planet0.mass/(xabs1)
    energies2 = planet0.mass*(vabs2**2)/2 - planet0.SolarSys.fconst*sun.mass*planet0.mass/(xabs2)
    # J = mrxv = r_x*v_y - r_y*v_x
    angmo1 = planet0.mass * np.cross(Eulerpositions, Eulervelocities)
    angmo2 = planet0.mass * np.cross(Leapfrogpositions, Leapfrogvelocities)

    ax1.set_title("Energy vs time")
    ax2.set_title("Angular Momentum vs time")

    ax1.plot(np.linspace(0, f, len(energies1)), energies1, label = "Euler")
    ax1.plot(np.linspace(0, f, len(energies2)), energies2, label = "Leapf")

    ax2.plot(np.linspace(0, f, len(angmo1)), angmo1, label = "Euler")
    ax2.plot(np.linspace(0, f, len(angmo2)), angmo2, label = "Leapf")

    
    ax1.legend()
    ax2.legend()
    txt = """
    Energy plots: The Euler method shows periodic variations (expected as the orbit is periodic hence the errors should be too)
    The leapfrog method shows almost no variation compared to the Euler method, as expected by conservation of energy.
    Angular momentum plots: Both methods agree well with conservation of angular momentum.
    """
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center')
    plt.show()

# two = input("Directions of initial velocity (press Enter to view): ")
# if two == "":
if True:
    pass

 
# # Instantiating of the solar system.
# SolarSys = SolarSystem ()

# # Instantiating of the sun.
# sun = Sun ( SolarSys )



# # Instantiating of planets.
# px = 200
# py = 0
# # v: for circular orbit at a given location
# v = np.sqrt(SolarSys.fconst*sun.mass/np.sqrt(px**2+py**2))
# theta = np.arctan(py/px)
# # delta = v*np.sqrt(2)-(np.cos(theta)-np.sin(theta))
# delta = 0.5
# sqrt2 = np.sqrt(2)

# planet0 = Planet ( SolarSys ,
#                     mass =10 ,
#                     position =(px , py) ,
#                     velocity =(-v*np.sin(theta) , v*np.cos(theta)),
#                     )
# # planet1 = Planet ( SolarSys ,
# #                     mass =10 ,
# #                     position =(px , py) ,
# #                     velocity =(-v*np.sin(theta) + delta, v*np.cos(theta)),
# #                     color = "tab:pink")
# # planet2 = Planet ( SolarSys ,
# #                     mass =10 ,
# #                     position =(px , py) ,
# #                     velocity =(-v*np.sin(theta) + delta/sqrt2, v*np.cos(theta) + delta/sqrt2),
# #                     color = "r")
# # planet3 = Planet ( SolarSys ,
# #                     mass =10 ,
# #                     position =(px , py) ,
# #                     velocity =(-v*np.sin(theta), v*np.cos(theta)+ delta), 
# #                     color = "tab:orange")
# # planet4 = Planet ( SolarSys ,
# #                     mass =10 ,
# #                     position =(px , py) ,
# #                     velocity =(-v*np.sin(theta) - delta/sqrt2, v*np.cos(theta) + delta/sqrt2), 
# #                     color = "tab:olive")
# # planet5 = Planet ( SolarSys ,
# #                     mass =10 ,
# #                     position =(px , py) ,
# #                     velocity =(-v*np.sin(theta) - delta, v*np.cos(theta)),
# #                     color = "g")
# # planet6 = Planet ( SolarSys ,
# #                     mass =10 ,
# #                     position =(px , py) ,
# #                     velocity =(-v*np.sin(theta) - delta/sqrt2, v*np.cos(theta) - delta/sqrt2), 
# #                     color = "tab:blue")
# # planet7 = Planet ( SolarSys ,
# #                     mass =10 ,
# #                     position =(px , py) ,
# #                     velocity =(-v*np.sin(theta), v*np.cos(theta) - delta),
# #                     color = "b")
# # planet8 = Planet ( SolarSys ,
# #                     mass =10 ,
# #                     position =(px , py) ,
# #                     velocity =(-v*np.sin(theta) + delta/sqrt2, v*np.cos(theta) - delta/sqrt2),
# #                     color = "tab:purple")

# # for trajectory skecthing purpose
# xpositions = {p:[] for p in SolarSys.planets}
# ypositions = {p:[] for p in SolarSys.planets}

# ### simulating planets travelling at parallel straight lines from far away
# # vx = -10
# # l = SolarSys.planets[::]
# # l.remove(SolarSys.get_planets("Sun"))
# # span = 800
# # for count, p in enumerate(l):
# #     p.velocity = (vx, 0)
# #     p.position = (1200, (count-len(l)/2+1)*span/len(l)*2)


# animate = True
# no_anim_frames = 500

# if animate:
#     # ## Animation
#     SolarSys.overwrite_update_to_quick(False)
#     def animate (i ):
#         """ This controls the animation."""
#         print (" The frame is : ", i)
#         # SolarSys.gravity_planets ()
#         # SolarSys.update_planets ()
        
#         SolarSys.update_planets () # -Leapfrog
#         SolarSys.gravity_planets () # -Leapfrog
#         SolarSys.update_planets () # -Leapfrog

#         SolarSys.fix_axes ()
#         # SolarSys.show_status()
        

#         for p in SolarSys.planets:
#             xpositions[p].append(p.position[0])
#             ypositions[p].append(p.position[1])

#         # SolarSys.show_status(['angmo', 'position', 'velocity'])

#     # This calls the animate function and creates animation.
#     anim = animation.FuncAnimation ( SolarSys.fig , animate ,
#     frames =100 , interval =50)

#     plt.show()

# else:
#     ## No animation:
#     SolarSys.overwrite_update_to_quick(True)
#     for i in range(no_anim_frames):
#         SolarSys.update_planets () # -Leapfrog
#         SolarSys.gravity_planets () # -Leapfrog
#         SolarSys.update_planets () # -Leapfrog

#         # SolarSys.fix_axes ()
#         # SolarSys.show_status()
        
#         for p in SolarSys.planets:
#             xpositions[p].append(p.position[0])
#             ypositions[p].append(p.position[1])

# # ### plotting tot energy and angmo against frame
# # for k, v in SolarSys.total_stats.items():
# #     fig, ax = plt.subplots()
# #     ax.plot(np.arange(0, len(v)),v)
# #     ax.set_title(k.capitalize())
# #     yrange = max(0.2*(max(v)-min(v)), 0.05*abs(max(v)+min(v))/2, 5)
# #     ax.set_ylim(min(v)-yrange, max(v)+yrange)
# #     plt.show()
# """

# ## plotting planet trajectories
# fig, ax = plt.subplots()

# for p in SolarSys.planets:
#     ax.plot(xpositions[p], ypositions[p], color = p.color)

# max_planet = planet0
# limrange = abs(max(xpositions[max_planet]+ypositions[max_planet],key = abs))*0.3
# # ax.set_xlim(( - limrange , limrange))
# # ax.set_ylim(( - limrange , limrange))
# # ax.set_xlim(( -  SolarSys.size /2, limrange))
# # ax.set_ylim(( - limrange , limrange))
# ax.set_xlim(( - SolarSys.size /2 , SolarSys.size /2))
# ax.set_ylim(( - SolarSys.size /2 , SolarSys.size /2))

# plt.show()


# # # This prepares the writer for the animation.
# # writervideo = animation.FFMpegWriter ( fps =60)
# # # This saves the animation.
# # anim.save (" planets_animation.mp4 ",
# # writer = writervideo , dpi =200)
# """