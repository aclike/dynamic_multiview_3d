from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
import math
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import os
import random


class Renderer(ShowBase):

    def __init__(self, depth, background, attenuation=True):

        self.generate_depth = depth
        self.replace_background = background
        self.attenuation = attenuation

        # Create flat textures / materials
        self.red_image = PNMImage(256, 256)
        self.red_image.fill(1, 0, 0)
        self.red_tex = Texture()
        self.red_tex.load(self.red_image)
        self.green_image = PNMImage(256, 256)
        self.green_image.fill(0, 1, 0)
        self.green_tex = Texture()
        self.green_tex.load(self.green_image)

        self.red_mat = Material()
        self.red_mat.setAmbient((1, 0, 0, 1))
        self.red_mat.setDiffuse((1, 0, 0, 1))
        self.red_mat.setEmission((1, 0, 0, 1))
        self.red_mat.setShininess(0.0)
        self.red_mat.setSpecular((1, 0, 0, 1))
        self.green_mat = Material()
        self.green_mat.setAmbient((0, 1, 0, 1))
        self.green_mat.setDiffuse((0, 1, 0, 1))
        self.green_mat.setEmission((0, 1, 0, 1))
        self.green_mat.setShininess(0.0)
        self.green_mat.setSpecular((0, 1, 0, 1))

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData('', 'win-size 128 128')
        ShowBase.__init__(self)
        base.disableMouse()
        base.setBackgroundColor(0.5, 0.5, 0.5)

        # setup scene
        self.scene = NodePath("Scene")
        self.scene.reparentTo(self.render)
        self.scene.setScale(1, 1, 1)
        self.scene.setTwoSided(True)
        self.scene.setPos(0, 0, 0)
        self.scene.setHpr(0, 0, 0)

        self.near_plane = 0.1
        self.far_plane = 5.0

        self.resolution = 128
        self.max_16bit_val = 65535

        self.light_sources = []
        self.light_nodes = []

        self.createLightSources()

        self.alight = AmbientLight('alight')
        self.alight.setColor(VBase4(10, 10, 10, 1))
        self.alnp = self.render.attachNewNode(self.alight)
        self.render.setLight(self.alnp)
        self.camera_target = None

        base.camLens.setNear(self.near_plane)
        base.camLens.setFar(self.far_plane)

        # prepare texture and camera for depth rendering
        if self.generate_depth is True:
            self.depth_tex = Texture()
            self.depth_tex.setFormat(Texture.FDepthComponent)
            self.depth_buffer = base.win.makeTextureBuffer(
                'depthmap', self.resolution, self.resolution,
                self.depth_tex, to_ram=True)
            self.depth_cam = self.makeCamera(self.depth_buffer,
                                             lens=base.camLens)
            print(self.depth_cam.node().getLens().getFilmSize())
            self.depth_cam.reparentTo(base.render)

        # list of models in memory
        self.models = []
        self.backgrounds = []
        self.model_positions = {}

    def delete(self):
        self.alnp.removeNode()
        for n in self.light_nodes:
            n.removeNode()
        for m in self.models:
            self.loader.unloadModel(m)
        base.destroy()

    def createLightSources(self):
        for i in range(0, 7):
            plight = PointLight('plight')
            if self.attenuation is True:
                plight.setAttenuation((1, 0, 1))
            plight.setColor(VBase4(0, 0, 0, 0))
            self.light_sources.append(plight)
            plnp = self.render.attachNewNode(plight)
            plnp.setPos(3, 3, 3)
            render.setLight(plnp)
            self.light_nodes.append(plnp)

    def activateLightSources(self, light_sources, spher=True):
        i = 0
        for lght in light_sources:
            lp_rad = lght[0]
            lp_el = lght[1]
            lp_az = lght[2]
            lp_int = lght[3]
            if spher:
                self.light_nodes[i].setPos(
                    lp_rad*math.cos(lp_el)*math.cos(lp_az),
                    lp_rad*math.cos(lp_el)*math.sin(lp_az),
                    lp_rad*math.sin(lp_el))
            else:
                self.light_nodes[i].setPos(lp_rad, lp_el, lp_az)
            self.light_sources[i].setColor(VBase4(lp_int, lp_int, lp_int, 1))
            i += 1

    def deactivateLightSources(self):
        for i in range(0, 7):
            self.light_sources[i].setColor(VBase4(0, 0, 0, 0))

    def textureToImage(self, texture):
        im = texture.getRamImageAs("RGB")
        strim = im.getData()
        image = np.fromstring(strim, dtype='uint8')
        image = image.reshape(self.resolution, self.resolution, 3)
        image = np.flipud(image)
        return image

    def textureToString(self, texture):
        im = texture.getRamImageAs("RGB")
        return im.getData()

    def setCameraPosition(self, rad, el, az, reuse_camera_target=False):
        xx = rad*math.cos(el)*math.cos(az)
        yy = rad*math.cos(el)*math.sin(az)
        zz = rad*math.sin(el)
        self.camera.setPos(xx, yy, zz)
        # print('camera position: %r' % ((xx, yy, zz,),))
        if self.camera_target is None or not reuse_camera_target:
            look_ind = min(self.model_positions.keys())
            self.camera_target = self.model_positions[look_ind]
        self.camera.lookAt(*self.camera_target)
        # print('looking at %r' % (self.model_positions[0],))

        if self.generate_depth is True:
            self.depth_cam.setPos(xx, yy, zz)
            self.depth_cam.lookAt(*self.camera_target)

    def loadImagenetBackgrounds(self, path, start, count):
        for i in range(start, start + count):
            fname = "ILSVRC2012_preprocessed_val_" + str(i).zfill(8) + ".JPEG"
            im = scipy.misc.imread(os.path.join(path, fname))
            im = scipy.misc.imresize(im,
                                     (self.resolution,
                                      self.resolution, 3), interp='nearest')
            self.backgrounds.append(im)

    @staticmethod
    def dist2(pos0, pos1):
        return math.sqrt((pos0[0] - pos1[0]) ** 2 + (pos0[1] - pos1[1]) ** 2)

    def showModel(self, model_ind, pos=None, yaw=None, debug=False, color=None):
        if pos is None:
            x_variation = 0.5
            y_variation = 2.0
            while True:
                pos = (random.random() * x_variation - x_variation * 0.5, random.random() * y_variation - y_variation * 0.5)
                for other_pos in self.model_positions.values():
                    distance = self.dist2(pos, other_pos)
                    if distance < 0.5 or distance > 1.5:
                        continue
                break
        if yaw is None:
            yaw = random.random() * 360
        if debug:
            print('position for model %d: %r' % (model_ind, pos))
        self.models[model_ind].reparentTo(self.scene)
        self.models[model_ind].setX(pos[0])
        self.models[model_ind].setY(pos[1])
        self.models[model_ind].setH(yaw)
        self.model_positions[model_ind] = self.models[model_ind].getPos()

        if color == 'red':
            self.models[model_ind].setTexture(self.red_tex)
            self.models[model_ind].setColor(1.0, 0.0, 0.0, 1.0, 1)  # r, g, b
            self.models[model_ind].setMaterial(self.red_mat)
        elif color == 'green':
            self.models[model_ind].setTexture(self.green_tex)
            self.models[model_ind].setColor(0.0, 1.0, 0.0, 1.0, 1)  # r, g, b
            self.models[model_ind].setMaterial(self.green_mat)

        return pos, yaw

    def hideModel(self, model_ind):
        # Cleanup
        self.models[model_ind].clearColor()
        self.models[model_ind].clearTexture()
        self.models[model_ind].clearMaterial()

        self.models[model_ind].detachNode()
        del self.model_positions[model_ind]

    def loadModels(self, model_names, models_path):
        mn = []
        for i in range(0, len(model_names)):
            mn.append(
                models_path + "/" +
                model_names[i].rstrip())

        self.models = self.loader.loadModel(mn)

    def renderView(self, camera_pos, light_sources,
                   blur, blending, spher=True, default_bg_setting=True, reuse_camera_target=False):

        self.setCameraPosition(camera_pos[0],
                               math.radians(camera_pos[1]),
                               math.radians(camera_pos[2]),
                               reuse_camera_target=reuse_camera_target)
        self.activateLightSources(light_sources, spher)

        base.graphicsEngine.renderFrame()
        tex = base.win.getScreenshot()
        im = self.textureToImage(tex)

        if self.generate_depth is True:
            depth_im = PNMImage()
            self.depth_tex.store(depth_im)

            depth_map = np.zeros([self.resolution,
                                  self.resolution], dtype='float')
            for i in range(0, self.resolution):
                for j in range(0, self.resolution):
                    depth_val = depth_im.getGray(j, i)
                    depth_map[i, j] = self.far_plane * self.near_plane /\
                        (self.far_plane - depth_val *
                            (self.far_plane - self.near_plane))
                    depth_map[i, j] = depth_map[i, j] / self.far_plane

            dm_uint = np.round(depth_map * self.max_16bit_val).astype('uint16')

        if self.replace_background is True and default_bg_setting is True:
            mask = (dm_uint == self.max_16bit_val)
            temp = np.multiply(
                mask.astype(dtype=np.float32).reshape(
                        self.resolution, self.resolution, 1), im)
            im = im - temp
            blurred_mask = scipy.ndimage.gaussian_filter(
                mask.astype(dtype=np.float32), blending)
            inv_mask = (blurred_mask - 1)*(-1)

            bg_ind = random.randint(0, len(self.backgrounds)-1)
            im = np.multiply(
                self.backgrounds[bg_ind],
                blurred_mask.reshape(self.resolution, self.resolution, 1)) + \
                np.multiply(im, inv_mask.reshape(self.resolution,
                                                 self.resolution, 1))

            im = scipy.ndimage.gaussian_filter(im, sigma=blur)

        im = im.astype(dtype=np.uint8)
        self.deactivateLightSources()

        return im, dm_uint
