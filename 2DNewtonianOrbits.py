import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
# from mpl_toolkits.mplot3d import Axes3D


def night():
    t = time.localtime().tm_hour
    if t<7 or t>21:
        return True
    else:
        return False
if night():
    plt.style.use('dark_background')

class SolarSystem () :
    """ This class creates the SolarSystem object."""
    def __init__ ( self ):
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
        self.fconst = 5
        self.dT = 0.5 # -Leapfrog
        # self.dT = 1
        
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
            planet.draw ()
    
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
        if len(targets)==1:
            return targets[0]
        elif len(targets)!=0:
            return targets
        else:
            return None # optional, returns None by default

    def show_status (self, stats_show = ["energy", "angmo"]):
        """stats_show: list of str(stats)"""
        su = self.get_planets("Sun")
        stats_show = [stat + "1f" if stat in ['position', 'velocity'] else stat for stat in stats_show]
        total_stats = {k:0 for k in stats_show if k not in ["position1f", "velocity1f"]}
        for p in self.planets:
            if p == su:
                continue
            stats = p.get_status()
            for s in total_stats.keys():
                total_stats[s] += stats[s]
            # display stats along with planets
            stats_tups       = [f"{k}: {v}" for (k, v) in stats.items() if (k in stats_show and type(v)==type(()))]
            stats_floats = [f"{k}: {v:.1f}" for (k, v) in stats.items() if (k in stats_show and isinstance(v, float))]
            self.ax.text(*p.position, "\n".join(stats_tups + stats_floats))
        for k, v in total_stats.items():
            self.total_stats[k].append(v)
        # display total energy and total angular momentum at the top right corner
        self.ax.text(self.size/5,
                    self.size/3, 
                    "\n".join([f"Total {k}: {v:.1f}" for (k,v) in total_stats.items() if k in ["energy", "angmo"]]))
    
            
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
        
    def move ( self ) :
        """ The planet is moved based on the velocity."""
        self.position = (
        self.position [0]+ self.velocity [0]* SolarSys.dT ,
        self.position [1]+ self.velocity [1]* SolarSys.dT #,
        # self.position [2]+ self.velocity [2]* SolarSys.dT #-3D
        )
        
    def draw ( self ) :
        """ The method to draw the planet."""
        self.SolarSys.ax.plot (* self.position ,
                                marker ="o",
                                markersize =10 ,
                                color = self.color
                                )
        
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
        for body in self , other :
            acceleration = np.divide ( force , body.mass )
            # acceleration = np.multiply ( acceleration , SolarSys.dT * switch )
            acceleration = np.multiply ( acceleration , SolarSys.dT * switch * 2) # -Leapfrog
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
        super ( Sun , self ). __init__ ( SolarSys , mass ,
        position , velocity )
        self.color = "y"
        
    def move ( self ) :
        self.position = self.position
        
    
# Instantiating of the solar system.
SolarSys = SolarSystem ()

# Instantiating of the sun.
sun = Sun ( SolarSys )

# Instantiating of planets.
px = 200
py = 0
# v: for circular orbit at a given location
v = np.sqrt(SolarSys.fconst*sun.mass/np.sqrt(px**2+py**2))
theta = np.arctan(py/px)
# delta = v*np.sqrt(2)-(np.cos(theta)-np.sin(theta))
delta = 0.5
sqrt2 = np.sqrt(2)

planet0 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta) , v*np.cos(theta)),
                    )
planet1 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta) + delta, v*np.cos(theta)),
                    color = "tab:pink")
planet2 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta) + delta/sqrt2, v*np.cos(theta) + delta/sqrt2),
                    color = "r")
planet3 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta), v*np.cos(theta)+ delta), 
                    color = "tab:orange")
planet4 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta) - delta/sqrt2, v*np.cos(theta) + delta/sqrt2), 
                    color = "tab:olive")
planet5 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta) - delta, v*np.cos(theta)),
                    color = "g")
planet6 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta) - delta/sqrt2, v*np.cos(theta) - delta/sqrt2), 
                    color = "tab:blue")
planet7 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta), v*np.cos(theta) - delta),
                    color = "b")
planet8 = Planet ( SolarSys ,
                    mass =10 ,
                    position =(px , py) ,
                    velocity =(-v*np.sin(theta) + delta/sqrt2, v*np.cos(theta) - delta/sqrt2),
                    color = "tab:purple")

# for trajectory skecthing purpose
xpositions = {p:[] for p in SolarSys.planets}
ypositions = {p:[] for p in SolarSys.planets}

### simulating planets travelling at parallel straight lines from far away
# vx = -10
# l = SolarSys.planets[::]
# l.remove(SolarSys.get_planets("Sun"))
# span = 800
# for count, p in enumerate(l):
#     p.velocity = (vx, 0)
#     p.position = (1200, (count-len(l)/2+1)*span/len(l)*2)


animate = True
no_anim_frames = 500

if animate:
    # ## Animation
    SolarSys.overwrite_update_to_quick(False)
    def animate (i ):
        """ This controls the animation."""
        print (" The frame is : ", i)
        # SolarSys.gravity_planets ()
        # SolarSys.update_planets ()
        
        SolarSys.update_planets () # -Leapfrog
        SolarSys.gravity_planets () # -Leapfrog
        SolarSys.update_planets () # -Leapfrog

        SolarSys.fix_axes ()
        # SolarSys.show_status()
        

        for p in SolarSys.planets:
            xpositions[p].append(p.position[0])
            ypositions[p].append(p.position[1])

        # SolarSys.show_status(['angmo', 'position', 'velocity'])

    # This calls the animate function and creates animation.
    anim = animation.FuncAnimation ( SolarSys.fig , animate ,
    frames =100 , interval =50)

    plt.show()

else:
    ## No animation:
    SolarSys.overwrite_update_to_quick(True)
    for i in range(no_anim_frames):
        SolarSys.update_planets () # -Leapfrog
        SolarSys.gravity_planets () # -Leapfrog
        SolarSys.update_planets () # -Leapfrog

        # SolarSys.fix_axes ()
        # SolarSys.show_status()
        
        for p in SolarSys.planets:
            xpositions[p].append(p.position[0])
            ypositions[p].append(p.position[1])

# ### plotting tot energy and angmo against frame
# for k, v in SolarSys.total_stats.items():
#     fig, ax = plt.subplots()
#     ax.plot(np.arange(0, len(v)),v)
#     ax.set_title(k.capitalize())
#     yrange = max(0.2*(max(v)-min(v)), 0.05*abs(max(v)+min(v))/2, 5)
#     ax.set_ylim(min(v)-yrange, max(v)+yrange)
#     plt.show()
"""

## plotting planet trajectories
fig, ax = plt.subplots()

for p in SolarSys.planets:
    ax.plot(xpositions[p], ypositions[p], color = p.color)

max_planet = planet0
limrange = abs(max(xpositions[max_planet]+ypositions[max_planet],key = abs))*0.3
# ax.set_xlim(( - limrange , limrange))
# ax.set_ylim(( - limrange , limrange))
# ax.set_xlim(( -  SolarSys.size /2, limrange))
# ax.set_ylim(( - limrange , limrange))
ax.set_xlim(( - SolarSys.size /2 , SolarSys.size /2))
ax.set_ylim(( - SolarSys.size /2 , SolarSys.size /2))

plt.show()


# # This prepares the writer for the animation.
# writervideo = animation.FFMpegWriter ( fps =60)
# # This saves the animation.
# anim.save (" planets_animation.mp4 ",
# writer = writervideo , dpi =200)
"""