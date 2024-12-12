import React from "react";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { FaCog } from "react-icons/fa";
import { EditorTheme, Settings } from "../types";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger } from "./ui/select";
import { capitalize } from "../lib/utils";
import { IS_RUNNING_ON_CLOUD } from "../config";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./ui/accordion";

interface Props {
  settings: Settings;
  setSettings: React.Dispatch<React.SetStateAction<Settings>>;
}

function SettingsDialog({ settings, setSettings }: Props) {
  const handleThemeChange = (theme: EditorTheme) => {
    setSettings((s) => ({
      ...s,
      editorTheme: theme,
    }));
  };

  return (
    <Dialog>
      <DialogTrigger>
        <FaCog />
      </DialogTrigger>
      <DialogContent
        className="max-w-full sm:max-w-md w-full p-4 sm:p-6 max-h-[90vh] overflow-y-auto"
      >
        <DialogHeader>
          <DialogTitle className="mb-4">Settings</DialogTitle>
        </DialogHeader>
  
        <div className="flex flex-col space-y-4">
          <div className="flex items-center space-x-2">
            <Label htmlFor="image-generation" className="flex-1">
              <div>DALL-E Placeholder Image Generation</div>
              <div className="font-light mt-2 text-xs">
                More fun with it but if you want to save money, turn it off.
              </div>
            </Label>
            <Switch
              id="image-generation"
              checked={settings.isImageGenerationEnabled}
              onCheckedChange={() =>
                setSettings((s) => ({
                  ...s,
                  isImageGenerationEnabled: !s.isImageGenerationEnabled,
                }))
              }
            />
          </div>
  
          <div>
            <Label htmlFor="openai-api-key">
              <div>OpenAI API key</div>
              <div className="font-light mt-1 mb-2 text-xs leading-relaxed">
                Only stored in your browser. Never stored on servers. Overrides
                your .env config.
              </div>
            </Label>
  
            <Input
              id="openai-api-key"
              placeholder="OpenAI API key"
              value={settings.openAiApiKey || ""}
              onChange={(e) =>
                setSettings((s) => ({
                  ...s,
                  openAiApiKey: e.target.value,
                }))
              }
            />
          </div>
  
          {!IS_RUNNING_ON_CLOUD && (
            <div>
              <Label htmlFor="openai-base-url">
                <div>OpenAI Base URL (optional)</div>
                <div className="font-light mt-2 leading-relaxed">
                  Replace with a proxy URL if you don't want to use the default.
                </div>
              </Label>
  
              <Input
                id="openai-base-url"
                placeholder="OpenAI Base URL"
                value={settings.openAiBaseURL || ""}
                onChange={(e) =>
                  setSettings((s) => ({
                    ...s,
                    openAiBaseURL: e.target.value,
                  }))
                }
              />
            </div>
          )}
  
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="item-1">
              <AccordionTrigger>Screenshot by URL Config</AccordionTrigger>
              <AccordionContent>
                <Label htmlFor="screenshot-one-api-key">
                  <div className="leading-normal font-normal text-xs">
                    If you want to use URLs directly instead of taking the
                    screenshot yourself, add a ScreenshotOne API key.{" "}
                    <a
                      href="https://screenshotone.com?via=screenshot-to-code"
                      className="underline"
                      target="_blank"
                    >
                      Get 100 screenshots/mo for free.
                    </a>
                  </div>
                </Label>
  
                <Input
                  id="screenshot-one-api-key"
                  className="mt-2"
                  placeholder="ScreenshotOne API key"
                  value={settings.screenshotOneApiKey || ""}
                  onChange={(e) =>
                    setSettings((s) => ({
                      ...s,
                      screenshotOneApiKey: e.target.value,
                    }))
                  }
                />
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
  
        <DialogFooter>
          <DialogClose>Save</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
  
}

export default SettingsDialog;
